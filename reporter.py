class Reporter:
    def __init__(self, filename):
        self.filename = filename

    def write_columns(self, columns):
        with open(self.filename, 'w') as file:
            file.write(",".join(columns))
            file.write("\n")

    def write_row(self, row):
        with open(self.filename, 'a') as file:
            file.write(",".join([f"{x}" for x in row]))
            file.write("\n")

