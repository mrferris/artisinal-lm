import pandas


def write_table(column_values):
    df_path = "benchmarks.csv"

    try:
        df = pandas.read_csv(df_path)
    except FileNotFoundError:
        df = pandas.DataFrame(columns=column_values.keys())

    df.loc[len(df)] = column_values.values()

    latex_table = df.groupby("model")[["mean", "std"]].mean().to_latex(float_format="%.3f")
    return latex_table
