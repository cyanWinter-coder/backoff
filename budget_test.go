package backoff

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestBudgetSharedTries verifies that all Retry calls under one budget draw
// from a single attempt pool: once it is empty, no member may start another
// attempt, and the group total never exceeds the limit.
func TestBudgetSharedTries(t *testing.T) {
	const maxTries = 7
	const workers = 5

	budget := NewBudget(WithBudgetMaxTries(maxTries))
	opErr := errors.New("transient")

	var calls atomic.Int64
	attempts := make([]atomic.Int64, workers)
	var wg sync.WaitGroup
	errs := make([]error, workers)
	for w := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, errs[w] = Retry(context.Background(),
				func() (bool, error) {
					calls.Add(1)
					attempts[w].Add(1)
					return false, opErr
				},
				WithBackOff(&ZeroBackOff{}), withTimer(&testTimer{}),
				WithBudget(budget),
			)
		}()
	}
	wg.Wait()

	if got := calls.Load(); got != maxTries {
		t.Errorf("total attempts = %d, want exactly %d across the group", got, maxTries)
	}
	for w, err := range errs {
		if !errors.Is(err, ErrBudgetExhausted) {
			t.Errorf("worker %d: Cause = %v, want ErrBudgetExhausted", w, err)
		}
		// A worker that ran at least one attempt keeps its last operation
		// error attached; one refused before its first attempt has none.
		if attempts[w].Load() > 0 {
			if !errors.Is(err, opErr) {
				t.Errorf("worker %d: operation error not in result: %v", w, err)
			}
		} else if re := AsRetryError(err); re == nil || re.LastErr != nil {
			t.Errorf("worker %d: LastErr = %v, want nil (no attempt ran)", w, err)
		}
	}
}

// TestBudgetExhaustedBeforeFirstAttempt verifies that an exhausted budget
// refuses even the first attempt: the operation never runs.
func TestBudgetExhaustedBeforeFirstAttempt(t *testing.T) {
	budget := NewBudget(WithBudgetMaxTries(1))

	// Drain the single shared try.
	_, err := Retry(context.Background(),
		func() (bool, error) { return false, errors.New("boom") },
		WithBackOff(&ZeroBackOff{}), withTimer(&testTimer{}),
		WithBudget(budget),
	)
	if !errors.Is(err, ErrBudgetExhausted) {
		t.Fatalf("first Retry: Cause = %v, want ErrBudgetExhausted", err)
	}

	calls := 0
	res, err := Retry(context.Background(),
		func() (int, error) { calls++; return 42, nil },
		WithBackOff(&ZeroBackOff{}), withTimer(&testTimer{}),
		WithBudget(budget),
	)
	if !errors.Is(err, ErrBudgetExhausted) {
		t.Errorf("second Retry: Cause = %v, want ErrBudgetExhausted", err)
	}
	if calls != 0 {
		t.Errorf("operation ran %d times against an exhausted budget, want 0", calls)
	}
	if res != 0 {
		t.Errorf("result = %d, want zero value when no attempt ran", res)
	}
	if re := AsRetryError(err); re == nil || re.LastErr != nil {
		t.Errorf("LastErr = %v, want nil when no attempt ran", err)
	}
	if strings.Contains(err.Error(), "%!") {
		t.Errorf("Error() contains a formatting artifact: %q", err.Error())
	}
}

// TestBudgetDeadline verifies that the shared deadline gates new attempts
// for every member of the group.
func TestBudgetDeadline(t *testing.T) {
	t.Run("already past", func(t *testing.T) {
		budget := NewBudget(WithBudgetDeadline(time.Now().Add(-time.Second)))
		calls := 0
		_, err := Retry(context.Background(),
			func() (bool, error) { calls++; return false, errors.New("boom") },
			withTimer(&testTimer{}), WithBudget(budget),
		)
		if !errors.Is(err, ErrBudgetExhausted) {
			t.Errorf("Cause = %v, want ErrBudgetExhausted", err)
		}
		if calls != 0 {
			t.Errorf("operation ran %d times past the deadline, want 0", calls)
		}
	})

	t.Run("expires between attempts", func(t *testing.T) {
		budget := NewBudget(WithBudgetMaxElapsedTime(time.Millisecond))
		opErr := errors.New("transient")
		calls := 0
		_, err := Retry(context.Background(),
			func() (bool, error) {
				calls++
				time.Sleep(5 * time.Millisecond) // outlive the shared deadline
				return false, opErr
			},
			WithBackOff(&ZeroBackOff{}), withTimer(&testTimer{}),
			WithBudget(budget),
		)
		if !errors.Is(err, ErrBudgetExhausted) {
			t.Errorf("Cause = %v, want ErrBudgetExhausted", err)
		}
		if !errors.Is(err, opErr) {
			t.Errorf("operation error not in result: %v", err)
		}
		if calls != 1 {
			t.Errorf("calls = %d, want 1 (the deadline gates the next attempt)", calls)
		}
	})
}

// TestBudgetConcurrentCountExact hammers the budget from many goroutines and
// verifies the final count is exact regardless of completion order: no lost
// or double-charged tries, no leaked slots. Run with -race.
func TestBudgetConcurrentCountExact(t *testing.T) {
	const maxTries = 50
	const workers = 32

	budget := NewBudget(WithBudgetMaxTries(maxTries), WithBudgetMaxConcurrent(4))

	var calls atomic.Int64
	var inFlight, maxInFlight atomic.Int64
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = Retry(context.Background(),
				func() (bool, error) {
					cur := inFlight.Add(1)
					for {
						old := maxInFlight.Load()
						if cur <= old || maxInFlight.CompareAndSwap(old, cur) {
							break
						}
					}
					calls.Add(1)
					inFlight.Add(-1)
					return false, errors.New("boom")
				},
				WithBackOff(&ZeroBackOff{}), withTimer(&testTimer{}),
				WithBudget(budget),
			)
		}()
	}
	wg.Wait()

	if got := calls.Load(); got != maxTries {
		t.Errorf("total attempts = %d, want exactly %d", got, maxTries)
	}
	if got := maxInFlight.Load(); got > 4 {
		t.Errorf("in-flight attempts peaked at %d, want at most 4", got)
	}
	// Every slot must have been returned: a fresh attempt can still be
	// refused by the try limit, but never block on a leaked slot.
	done := make(chan struct{})
	go func() {
		_, _ = Retry(context.Background(),
			func() (bool, error) { return true, nil },
			WithBudget(budget), withTimer(&testTimer{}),
		)
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Error("attempt after the group finished blocked on a leaked slot")
	}
}

// TestBudgetSlotReleasedDuringBackoff verifies that a Retry waiting out its
// backoff does not hold a concurrency slot: another member can run while the
// first is waiting.
func TestBudgetSlotReleasedDuringBackoff(t *testing.T) {
	budget := NewBudget(WithBudgetMaxConcurrent(1))

	const wait = 300 * time.Millisecond
	firstAttempted := make(chan struct{})
	var once sync.Once

	// A fails once and then waits out a long backoff before its second
	// (succeeding) attempt.
	aCalls := 0
	aDone := make(chan error, 1)
	go func() {
		_, err := Retry(context.Background(),
			func() (bool, error) {
				aCalls++
				once.Do(func() { close(firstAttempted) })
				if aCalls == 1 {
					return false, errors.New("boom")
				}
				return true, nil
			},
			WithBackOff(NewConstantBackOff(wait)),
			WithBudget(budget),
		)
		aDone <- err
	}()

	<-firstAttempted // A is now in its backoff wait, slot released.

	// B must acquire the only slot and finish well before A's wait ends.
	start := time.Now()
	bDone := make(chan error, 1)
	go func() {
		_, err := Retry(context.Background(),
			func() (bool, error) { return true, nil },
			WithBudget(budget),
		)
		bDone <- err
	}()

	select {
	case err := <-bDone:
		if err != nil {
			t.Errorf("B failed: %v", err)
		}
		if elapsed := time.Since(start); elapsed >= wait {
			t.Errorf("B completed after %v, want well under A's %v backoff wait", elapsed, wait)
		}
	case <-time.After(wait):
		t.Fatal("B blocked on a slot held by A's backoff wait")
	}

	if err := <-aDone; err != nil {
		t.Errorf("A failed: %v", err)
	}
}

// TestBudgetCloseWakesWaiters verifies that Close wakes members out of a
// backoff wait and out of a blocked slot acquisition, and refuses new joins.
func TestBudgetCloseWakesWaiters(t *testing.T) {
	t.Run("wakes a backoff wait", func(t *testing.T) {
		budget := NewBudget()
		opErr := errors.New("transient")
		attempted := make(chan struct{})
		var once sync.Once

		done := make(chan error, 1)
		go func() {
			_, err := Retry(context.Background(),
				func() (bool, error) {
					once.Do(func() { close(attempted) })
					return false, opErr
				},
				WithBackOff(NewConstantBackOff(time.Hour)), // real timer: only Close can wake it
				WithMaxElapsedTime(0),                      // don't let the 1h wait trip the elapsed limit
				WithBudget(budget),
			)
			done <- err
		}()

		<-attempted
		// Let Retry reach its wait before closing.
		time.Sleep(20 * time.Millisecond)
		budget.Close()

		select {
		case err := <-done:
			if !errors.Is(err, ErrBudgetClosed) {
				t.Errorf("Cause = %v, want ErrBudgetClosed", err)
			}
			if !errors.Is(err, opErr) {
				t.Errorf("operation error not in result: %v", err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("Close did not wake the backoff wait")
		}
	})

	t.Run("wakes a blocked slot acquisition", func(t *testing.T) {
		budget := NewBudget(WithBudgetMaxConcurrent(1))
		release := make(chan struct{})
		attempted := make(chan struct{})

		// A holds the only slot inside its operation.
		aDone := make(chan error, 1)
		go func() {
			_, err := Retry(context.Background(),
				func() (bool, error) {
					close(attempted)
					<-release
					return true, nil
				},
				WithBudget(budget),
			)
			aDone <- err
		}()
		<-attempted

		// B blocks acquiring the slot; Close must wake it without running op.
		bCalls := 0
		bDone := make(chan error, 1)
		go func() {
			_, err := Retry(context.Background(),
				func() (bool, error) { bCalls++; return true, nil },
				WithBudget(budget),
			)
			bDone <- err
		}()

		time.Sleep(20 * time.Millisecond) // let B block on the slot
		budget.Close()

		select {
		case err := <-bDone:
			if !errors.Is(err, ErrBudgetClosed) {
				t.Errorf("B: Cause = %v, want ErrBudgetClosed", err)
			}
			if bCalls != 0 {
				t.Errorf("B ran %d attempts after close, want 0", bCalls)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("Close did not wake the blocked slot acquisition")
		}

		close(release)
		if err := <-aDone; err != nil {
			t.Errorf("A failed: %v", err)
		}
	})

	t.Run("refuses new joins", func(t *testing.T) {
		budget := NewBudget()
		budget.Close()
		calls := 0
		_, err := Retry(context.Background(),
			func() (bool, error) { calls++; return true, nil },
			WithBudget(budget), withTimer(&testTimer{}),
		)
		if !errors.Is(err, ErrBudgetClosed) {
			t.Errorf("Cause = %v, want ErrBudgetClosed", err)
		}
		if calls != 0 {
			t.Errorf("operation ran %d times on a closed budget, want 0", calls)
		}
	})
}

// TestBudgetCloseIdempotent verifies repeated Close calls and a budget that
// never saw an operation neither panic nor deadlock.
func TestBudgetCloseIdempotent(t *testing.T) {
	budget := NewBudget(WithBudgetMaxTries(1), WithBudgetMaxConcurrent(1))
	budget.Close()
	budget.Close() // must not panic

	select {
	case <-budget.Done():
	default:
		t.Error("Done() not closed after Close")
	}

	// Closing a budget nobody used must be a no-op.
	NewBudget().Close()
}

// TestBudgetCausesPreserved verifies that attaching a budget does not change
// how per-operation outcomes are reported: success, Permanent, RetryAfter,
// and context cancellation behave exactly as without one.
func TestBudgetCausesPreserved(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		res, err := Retry(context.Background(),
			func() (int, error) { return 42, nil },
			WithBudget(NewBudget()), withTimer(&testTimer{}),
		)
		if err != nil || res != 42 {
			t.Errorf("Retry = (%d, %v), want (42, nil)", res, err)
		}
	})

	t.Run("permanent", func(t *testing.T) {
		opErr := errors.New("nope")
		_, err := Retry(context.Background(),
			func() (bool, error) { return false, Permanent(opErr) },
			WithBudget(NewBudget()), withTimer(&testTimer{}),
		)
		if !errors.Is(err, ErrPermanent) {
			t.Errorf("Cause = %v, want ErrPermanent", err)
		}
		if !errors.Is(err, opErr) {
			t.Errorf("operation error not in result: %v", err)
		}
	})

	t.Run("retry after", func(t *testing.T) {
		cause := errors.New("rate limited")
		tm := &spyTimer{}
		calls := 0
		_, err := Retry(context.Background(),
			func() (int, error) {
				calls++
				if calls == 1 {
					return 0, RetryAfter(42*time.Second, cause)
				}
				return 1, nil
			},
			WithBudget(NewBudget()), withTimer(tm), WithMaxElapsedTime(0),
		)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(tm.starts) != 1 || tm.starts[0] != 42*time.Second {
			t.Errorf("waits = %v, want a single wait of 42s", tm.starts)
		}
	})

	t.Run("context cancelled during wait", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		calls := 0
		_, err := Retry(ctx,
			func() (bool, error) {
				calls++
				cancel()
				return false, errors.New("boom")
			},
			WithBudget(NewBudget()),
			WithBackOff(&ZeroBackOff{}), withTimer(&testTimer{}),
		)
		if !errors.Is(err, context.Canceled) {
			t.Errorf("Cause = %v, want context.Canceled", err)
		}
	})

	t.Run("context cancelled while blocked on a slot", func(t *testing.T) {
		budget := NewBudget(WithBudgetMaxConcurrent(1))
		release := make(chan struct{})
		defer close(release)
		attempted := make(chan struct{})

		go func() {
			_, _ = Retry(context.Background(),
				func() (bool, error) { close(attempted); <-release; return true, nil },
				WithBudget(budget),
			)
		}()
		<-attempted

		ctx, cancel := context.WithCancel(context.Background())
		done := make(chan error, 1)
		go func() {
			_, err := Retry(ctx,
				func() (bool, error) { return true, nil },
				WithBudget(budget),
			)
			done <- err
		}()
		time.Sleep(20 * time.Millisecond) // let the second Retry block on the slot
		cancel()

		select {
		case err := <-done:
			if !errors.Is(err, context.Canceled) {
				t.Errorf("Cause = %v, want context.Canceled", err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("cancellation did not wake the blocked slot acquisition")
		}
	})
}

// TestBudgetNil verifies that WithBudget(nil) leaves Retry unchanged.
func TestBudgetNil(t *testing.T) {
	calls := 0
	_, err := Retry(context.Background(),
		func() (bool, error) {
			calls++
			if calls < 3 {
				return false, fmt.Errorf("attempt %d", calls)
			}
			return true, nil
		},
		WithBudget(nil), WithBackOff(&ZeroBackOff{}), withTimer(&testTimer{}),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if calls != 3 {
		t.Errorf("calls = %d, want 3", calls)
	}
}
