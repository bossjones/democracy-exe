# HTML Slider Output and Charts Feature

## High-Level Objective

- Create an interactive HTML output format with dynamic word frequency visualization

## Mid-Level Objective

- Build new HTML output format with slider-based word frequency filtering
- Add radial bar and bubble chart visualizations for word frequencies
- Extend CLI to support new output formats
- Ensure comprehensive test coverage

## Implementation Notes
- No need to import any external libraries beyond existing dependencies
- Comment every new function
- For CLI commands add usage examples starting with `uv run main.py`
- Follow existing code patterns and type safety practices
- Add tests for all new functionality

## Context

### Beginning context
- `src/let_the_code_write_itself/output_format.py`
- `src/let_the_code_write_itself/chart.py`
- `src/let_the_code_write_itself/main.py`
- `src/let_the_code_write_itself/tests/output_format_test.py`
- `src/let_the_code_write_itself/tests/chart_test.py`

### Ending context
- `src/let_the_code_write_itself/output_format.py`
- `src/let_the_code_write_itself/chart.py`
- `src/let_the_code_write_itself/main.py`
- `src/let_the_code_write_itself/tests/output_format_test.py`
- `src/let_the_code_write_itself/tests/chart_test.py`

## Low-Level Tasks
> Ordered from start to finish

1. Add new HTML output format with slider
```aider
UPDATE src/let_the_code_write_itself/output_format.py:
    CREATE format_as_html_with_slider_filter() function:
        Add HTML template with slider control
        Add JavaScript for dynamic filtering
        MIRROR format_as_html()
```

2. Add new chart visualizations
```aider
UPDATE src/let_the_code_write_itself/chart.py:
    CREATE create_radial_bar_chart(word_counts: WordCounts), create_bubble_chart(...)
```

3. Update CLI interface
```aider
UPDATE src/let_the_code_write_itself/main.py:
    ADD support for checking .htmlsld extension and calling format_as_html_with_slider_filter()
        Be sure to use .html when saving the file, .htmlsld is just for checking
    ADD support for 'radial' and 'bubble' choices and calling respective chart functions

```

4. Add comprehensive tests
```aider
UPDATE test files:
    ADD test_format_as_html_with_slider_filter()
    ADD test_create_radial_bar_chart()
    ADD test_create_bubble_chart()
```
