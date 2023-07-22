from pydantic import BaseModel


class SolveCaptchaResponse(BaseModel):
    solution: int
