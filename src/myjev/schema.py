from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_INPUT_BYTES = 256 * 1024
MAX_HTTP_BODY_BYTES = 1024 * 1024


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
        # Bound preprocessing independently of the exact tokenizer-token limit.
        fields = [self.context, self.instructions]
        fields.extend(value for c in self.candidates for value in (c.id, c.description))
        if sum(len(value.encode("utf-8")) for value in fields) > MAX_INPUT_BYTES:
            raise ValueError(f"combined request text exceeds {MAX_INPUT_BYTES} UTF-8 bytes")
        ids = [c.id for c in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate IDs must be unique")
        return self
