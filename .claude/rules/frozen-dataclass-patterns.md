# Frozen Dataclass Patterns

## Rules

- All configuration types are `@dataclass(frozen=True)` — ReviewProfile, PlanProfile, QualityLoopSpec, ScorePolicy
- Lightweight criterion types use `NamedTuple` — ReviewCriterion, PlanCriterion, HardBlocker
- Add new fields with defaults to maintain backward compatibility with existing profile instances
- Never mutate profile instances; use `dataclasses.replace()` to create variants
- Use `tuple` for immutable sequences, never `list`, in frozen dataclasses
- `state_field_aliases` dict handles field name versioning for backward compat

## Example

```python
class MyCriterion(NamedTuple):
    name: str
    weight: int
    description: str = ""

@dataclass(frozen=True)
class MyProfile:
    system_id: str
    criteria: tuple[MyCriterion, ...] = ()
    threshold: int = 90
    # New fields always have defaults
    new_option: bool = False
```
