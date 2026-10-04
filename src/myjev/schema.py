from pydantic import BaseModel, ConfigDict, Field, model_validator


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=8192)


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    context: str = Field(max_length=100000)
    instructions: str = Field(min_length=1, max_length=8192)
    candidates: list[Candidate] = Field(min_length=2, max_length=160)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [c.id for c in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate IDs must be unique")
        return self
