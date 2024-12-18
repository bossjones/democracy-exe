from __future__ import annotations

import marimo as mo


app = mo.App()


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __(mo):
    # Create a title for our interactive demo
    mo.md("""
    # Interactive Number Selector
    Choose a number between 0 and 4 using either the slider or dropdown menu.
    """)


@app.cell
def __(mo):
    # Create a slider for number selection
    slider = mo.ui.slider(start=0, stop=4, step=1, value=2, label="Select a number with slider")
    return (slider,)


@app.cell
def __(mo):
    # Create a dropdown for number selection
    dropdown = mo.ui.dropdown(options=["0", "1", "2", "3", "4"], value="2", label="Select a number from dropdown")
    return (dropdown,)


@app.cell
def __(mo, slider, dropdown):
    # Display the components in a vertical stack
    mo.vstack([
        slider,
        mo.md("---"),  # Add a separator
        dropdown,
        mo.md("---"),  # Add another separator
        mo.md(f"""
        ### Selected Values
        - Slider value: {slider}
        - Dropdown value: {dropdown}
        """),
    ])
