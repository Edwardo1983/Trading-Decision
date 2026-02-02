class RateLimiter:
    def __init__(self):
        self.weight = 0

    def add(self, weight: int) -> None:
        self.weight += weight

    def reset(self) -> None:
        self.weight = 0