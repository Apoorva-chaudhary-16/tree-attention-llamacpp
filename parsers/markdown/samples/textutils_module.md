# textutils Module

The `textutils` module provides small helpers for cleaning text. It requires Python 3.9 or newer.

## Functions

### slugify
```python
def slugify(text: str, sep: str = "-") -> str:
    """Convert text to a URL friendly slug."""
```
Lowercases the text, removes accents and replaces spaces with the separator. For example, `slugify("Hello World")` returns `hello-world`.

### truncate
```python
def truncate(text: str, width: int = 80) -> str:
    """Shorten text to at most width characters."""
```
The default width is 80. Truncated text ends with three dots.

### word_count
```python
def word_count(text: str) -> int:
    """Count words separated by whitespace."""
```
Returns the number of words in the text.

## Classes

### Cleaner
The Cleaner class applies several steps in order. It has the methods `add_step`, `run` and `reset`. A new Cleaner starts with no steps.

## Exceptions
`EmptyTextError` is raised when an empty string is passed to `slugify`.

## License
The module is released under the MIT license.
