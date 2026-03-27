# VTM Usage

Currently vtm is scoped for being used with agents written in python, this may be expanded within the future.

VTM is compatable with any projects and repo in any coding language supporter by treesitter, so while your agent is written in python, your agent and memory can be using on javascript, c++, etc. repos or multilanguage repos.

In order to use vtm you should install the vtm package from pip, uv, etc. (obviously will get more flushed out here)

blah blah blah

now import all of vtm with `from vtm import *` or important just the componenets you need from `from vtm important memory, ...` check the remaining docs in struct and types for more info on what vtm does and exposes.

NOTE: a lot of vtm types rely on each other so will need to import them together (check and fact check how this works)

## Getting Started

Let's first get a mental model of how to use VTM.

Use VTM when the agent learns something that may matter later and should be reusable safely.
The most common cases are:
- a verified claim about a bug or behavior
- a procedure for re-checking a condition
- a tool-produced artifact worth reusing
- a code-localized observation tied to an anchor

It is obviously upto you if you would like to allow LLM agents to do this directly themselves or have a human in the loop, etc.

### Typical write flow

A normal write flow looks like this:
1. produce or observe an artifact
2. create any relevant code anchor
3. assemble a `Memory`
4. attach evidence references
5. stage or persist the memory
6. verify now or later depending on policy

### Example: writing a claim memory

```python
from vtm.types import Memory, MemoryType, VerifyStatus

memory = Memory(
    memory_id="mem:claim:empty-input-parse",
    memory_type=MemoryType.CLAIM,
    evidence=[
        {"kind": "artifact_ref", "artifact_id": "artifact:test-log:9c2b1d", "...": "..."},
        {"kind": "code_anchor", "path": "pkg/parser.py", "symbol": "parse", "...": "...",},
    ],
    status=VerifyStatus.UNVERIFIED,
    verifier_script="verifiers/check_parse_empty.py",
)
```
