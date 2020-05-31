def markdown_to_string(file_path):
    with open(file_path, 'r') as f:
        string = f.read()

    return string