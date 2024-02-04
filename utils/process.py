from pathlib import Path


def process_args(args: dict) -> tuple[list[str], Path]:
    path = Path("runtime_store/config.toml")
    if not path.exists():
        path.touch()
    output_args = []
    with path.open(mode="w", encoding="utf-8") as f:
        for key, value in args.items():
            if isinstance(value, str):
                value = f'"{value}"'
            if isinstance(value, bool):
                value = f"{value}".lower()
            to_print = f"{key} = {value}"
            output_args.append(to_print)
            f.write(to_print + "\n")
    return output_args, path


def process_dataset_args(args: dict) -> tuple[dict, Path]:
    path = Path("runtime_store/dataset.toml")
    if not path.exists():
        path.touch()
    output_args = {}
    with path.open(mode="w", encoding="utf-8") as f:
        output_args["general"] = []
        f.write("[general]\n")
        for key, value in args["general"].items():
            if isinstance(value, str):
                value = f'"{value}"'
            if isinstance(value, bool):
                value = f"{value}".lower()
            to_print = f"{key} = {value}"
            output_args["general"].append(to_print)
            f.write(to_print + "\n")
        f.write("\n[[datasets]]\n")
        output_args["subsets"] = []
        for subset in args["subsets"]:
            f.write("\n\t[[datasets.subsets]]\n")
            output_args["subsets"].append([])
            for key, value in subset.items():
                if isinstance(value, str):
                    value = f'"{value}"'
                if isinstance(value, bool):
                    value = f"{value}".lower()
                to_print = f"{key} = {value}"
                output_args["subsets"][-1].append(to_print)
                f.write(f"\t{to_print}\n")
    return output_args, path
