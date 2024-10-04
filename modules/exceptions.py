class NotFoundError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class VectorizerError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(f"Error during vectorization process: {msg}")


class DatabaseError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)
