class Reporter:
    @staticmethod
    def get_filename():
        return "test.csv"

    @staticmethod
    def write_columns(columns):
        with open(Reporter.get_filename(), 'w') as file:
            file.write(",".join(columns))
            file.write("\n")

    @staticmethod
    def write_row(row):
        with open(Reporter.get_filename(), 'a') as file:
            file.write(",".join([f"{x}" for x in row]))
            file.write("\n")

