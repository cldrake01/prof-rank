def render(bio: str, n: int = 80) -> None:
    """
    Display the bio within tje width of n characters.
    Words are carried over to the next line.
    This is done by printing word-by-word, adding a newline
    when the current line exceeds the width.
    """
    words = bio.split()
    line = ""
    for word in words:
        if len(line) + len(word) + 1 <= n:
            line += word + " "
        else:
            print(line)
            line = word + " "
    print(line)
