package backoff

import (
	"context"
	"sync"
	"time"
)

// Budget is a retry budget shared by a group of concurrent [Retry] calls.
// Attach it with [WithBudget] to bound a whole group of operations instead of
// each operation separately: without one, every [Retry] call tracks only its
// own attempts and waits, so a failure storm across many downstream probes
// multiplies the pause window by the number of callers.
//
// A budget pools three things across every [Retry] that shares it:
//
//   - a total attempt count ([WithBudgetMaxTries]): every attempt started by
//     any member of the group consumes one shared try, and once the pool is
//     empty no member may start another attempt;
//   - a total deadline ([WithBudgetMaxElapsedTime] or [WithBudgetDeadline]):
//     no attempt may start after it passes;
//   - a concurrency limit ([WithBudgetMaxConcurrent]): how many attempts may
//     be in flight at once. A slot is held only while the operation runs and
//     is released before the backoff wait, so waiting operations never block
//     others from trying.
//
// Consumption is atomic: a try is deducted and a slot taken in a single
// locked step, only when the attempt is certain to start. A [Retry] that
// loses the race for the last try is never charged for it, and a released
// slot is never returned twice.
//
// When the budget is exhausted, closed, or past its deadline, [Retry] stops
// before the next attempt and returns a [*RetryError] with Cause
// [ErrBudgetExhausted] or [ErrBudgetClosed]; the last operation error is
// still reported as LastErr. Each member of the group keeps its own result,
// backoff policy, and per-call limits ([WithMaxTries], [WithMaxElapsedTime])
// — the budget only gates whether an attempt may start.
//
// A Budget is safe for concurrent use. Copying a Budget after first use is
// not allowed.
type Budget struct {
	mu            sync.Mutex
	maxTries      uint          // shared attempt limit; 0 means unlimited
	tries         uint          // attempts started across the whole group
	deadline      time.Time     // shared wall-clock deadline; zero means none
	maxConcurrent uint          // in-flight attempt limit; 0 means unlimited
	slots         chan struct{} // in-flight attempt slots; nil means unlimited
	done          chan struct{} // closed by Close
	closed        bool
}

// BudgetOption configures a [Budget].
type BudgetOption func(*Budget)

// NewBudget returns a [Budget] that can be shared by any number of [Retry]
// calls via [WithBudget]. By default the budget imposes no limits and only
// provides group-wide [Budget.Close] semantics.
func NewBudget(opts ...BudgetOption) *Budget {
	b := &Budget{done: make(chan struct{})}
	for _, opt := range opts {
		opt(b)
	}
	if b.maxConcurrent > 0 {
		b.slots = make(chan struct{}, b.maxConcurrent)
	}
	return b
}

// WithBudgetMaxTries limits the total number of attempts started by all
// [Retry] calls sharing the budget, not per call: WithBudgetMaxTries(10)
// allows ten attempts across the whole group, however they are distributed.
// When the pool is empty, further attempts are refused with
// [ErrBudgetExhausted]. The default, 0, means no shared limit.
func WithBudgetMaxTries(n uint) BudgetOption {
	return func(b *Budget) {
		b.maxTries = n
	}
}

// WithBudgetMaxElapsedTime sets a shared wall-clock deadline d from now. No
// new attempt may start after it passes; attempts refused because of it
// report [ErrBudgetExhausted]. The deadline is checked when an attempt
// starts: it interrupts neither an in-flight attempt nor a backoff wait in
// progress. The default, 0, means no shared deadline.
func WithBudgetMaxElapsedTime(d time.Duration) BudgetOption {
	return func(b *Budget) {
		if d > 0 {
			b.deadline = time.Now().Add(d)
		}
	}
}

// WithBudgetDeadline sets a shared absolute deadline. See
// [WithBudgetMaxElapsedTime] for how the deadline is enforced.
func WithBudgetDeadline(t time.Time) BudgetOption {
	return func(b *Budget) {
		b.deadline = t
	}
}

// WithBudgetMaxConcurrent limits how many attempts under the budget may run
// at once. A slot is held only for the duration of the operation call itself
// and released before the backoff wait begins, so a group member waiting out
// its backoff never occupies a slot. An attempt that cannot take a slot
// blocks until one frees up, the budget is closed, or the [Retry] context is
// cancelled. The default, 0, means no concurrency limit.
func WithBudgetMaxConcurrent(n uint) BudgetOption {
	return func(b *Budget) {
		b.maxConcurrent = n
	}
}

// Done returns a channel closed when [Budget.Close] is called. [Retry] uses
// it to wake members of the group out of their backoff wait.
func (b *Budget) Done() <-chan struct{} { return b.done }

// Close shuts the budget down: every [Retry] waiting out a backoff or blocked
// on a concurrency slot wakes immediately and stops with [ErrBudgetClosed],
// and no new attempt may start. Attempts already running are not interrupted.
// Close is idempotent and safe to call on a budget no [Retry] ever used.
func (b *Budget) Close() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.closed {
		b.closed = true
		close(b.done)
	}
}

// budgetPermit is a granted attempt: one shared try consumed, plus a
// concurrency slot while the operation runs. It is released exactly once.
type budgetPermit struct {
	b        *Budget
	released sync.Once
}

// release returns the concurrency slot. The consumed try is never refunded:
// the attempt happened, successful or not. Release is idempotent.
func (p *budgetPermit) release() {
	p.released.Do(func() {
		if p.b.slots != nil {
			<-p.b.slots
		}
	})
}

// acquire grants one attempt under the budget. It blocks while the
// concurrency limit is reached and fails without consuming a try when the
// budget is closed, exhausted, past its deadline, or ctx is cancelled.
//
// The try is consumed and the slot taken in one locked step, after every
// refusal check has passed, so a lost race neither charges the caller nor
// leaves a slot to be returned twice.
func (b *Budget) acquire(ctx context.Context) (*budgetPermit, error) {
	// Take a concurrency slot first: it is the only step that can block, and
	// nothing is consumed while waiting, so a close or cancellation that
	// loses the race has nothing to refund.
	if b.slots != nil {
		select {
		case b.slots <- struct{}{}:
		case <-b.done:
			return nil, ErrBudgetClosed
		case <-ctx.Done():
			return nil, context.Cause(ctx)
		}
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	switch {
	case b.closed:
		b.releaseSlot()
		return nil, ErrBudgetClosed
	case b.maxTries > 0 && b.tries >= b.maxTries:
		b.releaseSlot()
		return nil, ErrBudgetExhausted
	case !b.deadline.IsZero() && !time.Now().Before(b.deadline):
		b.releaseSlot()
		return nil, ErrBudgetExhausted
	}
	b.tries++
	return &budgetPermit{b: b}, nil
}

// releaseSlot returns a slot taken before the refusal checks failed. Only
// called on paths where no permit was granted, so it cannot pair with a
// later permit release.
func (b *Budget) releaseSlot() {
	if b.slots != nil {
		<-b.slots
	}
}

// WithBudget attaches a shared [Budget] to [Retry]: every attempt — including
// the first — must be granted by the budget before it starts, and the group
// stops together when the budget is exhausted or closed. See [Budget] for the
// shared semantics. A nil budget disables the option.
//
// The budget gates attempts only; everything else about [Retry] is unchanged.
// Each call keeps its own backoff policy, result, and error, and the
// [ErrPermanent], [RetryAfterError], and context-cancellation causes are
// reported exactly as without a budget.
func WithBudget(b *Budget) RetryOption {
	return func(args *retryOptions) {
		args.Budget = b
	}
}
