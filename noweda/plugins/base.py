class BasePlugin:
    name = "base"

    def run(self, df):
        raise NotImplementedError
