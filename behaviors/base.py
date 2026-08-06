"""行为基类。每个状态行为实现 enter/step/exit；step 返回 (cmd, next_mode)。"""


class Behavior:
    name = ''

    def enter(self, ctx):  # noqa: B027
        pass

    def step(self, ctx, now):  # noqa: B027
        raise NotImplementedError

    def exit(self, ctx):  # noqa: B027
        pass
