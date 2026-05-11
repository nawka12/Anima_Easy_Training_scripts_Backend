from pathlib import Path
import json

def process_args(args: dict) -> tuple[list[str], Path]:
    path = Path("runtime_store/config.toml")
    if not path.exists():
        path.touch()
    output_args = []
    with path.open(mode="w", encoding="utf-8") as f:
        for key, value in args.items():
            formatted_value = json.dumps(value)

            to_print = f"{key} = {formatted_value}"
            output_args.append(to_print)
            f.write(to_print + "\n")
    return output_args, path


def _write_subset_block(f, subset: dict, output_lines: list) -> None:
    f.write("\n\t[[datasets.subsets]]\n")
    for key, value in subset.items():
        formatted_value = json.dumps(value)
        to_print = f"{key} = {formatted_value}"
        output_lines.append(to_print)
        f.write(f"\t{to_print}\n")


def process_dataset_args(args: dict) -> tuple[dict, Path]:
    path = Path("runtime_store/dataset.toml")
    if not path.exists():
        path.touch()

    # Multi-dataset (multi-resolution) mode: each entry in args["datasets"]
    # carries its own dataset-level keys plus its subsets, and we emit one
    # [[datasets]] block per entry with no top-level [general] section.
    if "datasets" in args:
        output_args = {"datasets": []}
        with path.open(mode="w", encoding="utf-8") as f:
            for i, dataset in enumerate(args["datasets"]):
                dataset_general = dataset.get("general", {})
                dataset_subsets = dataset.get("subsets", [])
                dataset_out = {"general": [], "subsets": []}

                if i > 0:
                    f.write("\n")
                f.write("[[datasets]]\n")
                for key, value in dataset_general.items():
                    formatted_value = json.dumps(value)
                    to_print = f"{key} = {formatted_value}"
                    dataset_out["general"].append(to_print)
                    f.write(f"{to_print}\n")

                for subset in dataset_subsets:
                    subset_lines = []
                    _write_subset_block(f, subset, subset_lines)
                    dataset_out["subsets"].append(subset_lines)

                output_args["datasets"].append(dataset_out)
        return output_args, path

    # Legacy single-dataset shape: [general] + one [[datasets]] block.
    output_args = {"general": [], "subsets": []}
    with path.open(mode="w", encoding="utf-8") as f:
        f.write("[general]\n")
        for key, value in args["general"].items():
            formatted_value = json.dumps(value)
            to_print = f"{key} = {formatted_value}"
            output_args["general"].append(to_print)
            f.write(to_print + "\n")

        f.write("\n[[datasets]]\n")
        for subset in args["subsets"]:
            subset_lines = []
            _write_subset_block(f, subset, subset_lines)
            output_args["subsets"].append(subset_lines)
    return output_args, path
