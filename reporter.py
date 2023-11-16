class Reporter:
    def __init__(self, dwt=True,indexify="sigmoid", retain_relative_position=True,random_initialize=True):
        self.dwt = dwt
        self.indexify = indexify
        self.retain_relative_position = retain_relative_position
        self.random_initialize = random_initialize

    def get_filename(self):
        return f"{str(self.dwt)}_{self.indexify}_{str(self.retain_relative_position)}_{str(self.random_initialize)}.csv"

    def write_columns(self, columns):
        with open(self.get_filename(), 'w') as file:
            file.write(",".join(columns))
            file.write("\n")

    def write_row(self, row):
        with open(self.get_filename(), 'a') as file:
            file.write(",".join([f"{x}" for x in row]))
            file.write("\n")

