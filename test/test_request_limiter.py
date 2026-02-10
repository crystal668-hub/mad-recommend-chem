import threading
import time
import unittest


class RequestLimiterTests(unittest.TestCase):
    def test_concurrency_cap_and_no_deadlock(self):
        from utils.request_limiter import GlobalRequestLimiter

        cap = 3
        limiter = GlobalRequestLimiter(max_inflight=cap)

        lock = threading.Lock()
        inflight = 0
        max_seen = 0

        def worker():
            nonlocal inflight, max_seen
            with limiter.slot("llm"):
                with lock:
                    inflight += 1
                    if inflight > max_seen:
                        max_seen = inflight
                time.sleep(0.05)
                with lock:
                    inflight -= 1

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertFalse(any(t.is_alive() for t in threads), "threads should not deadlock")
        self.assertLessEqual(max_seen, cap, "inflight concurrency should be capped")
        self.assertEqual(limiter.inflight, 0, "limiter should have no leaked slots")


if __name__ == "__main__":
    unittest.main()

