import json
import os


def filter_merge_sort_json(file1, file2, output_file):
    all_data = []

    try:
        with open(file1, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)
    except FileNotFoundError:
        print(f"Error: File {file1} does not exist!")
        return
    except json.JSONDecodeError:
        print(f"Error: File {file1} is not a valid JSON!")
        return

    try:
        with open(file2, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                filtered_data = [
                    item for item in data if item.get("initial_correct") is True
                ]
                all_data.extend(filtered_data)
            else:
                print(f"Error: The data format in file {file2} is incorrect!")
                return
    except FileNotFoundError:
        print(f"Error: File {file2} does not exist!")
        return
    except json.JSONDecodeError:
        print(f"Error: File {file2} is not a valid JSON!")
        return

    sorted_data = sorted(all_data, key=lambda x: x["id"])

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=4)
    print(f"Merge, filter, and sort completed! Results saved to: {output_file}")


def process_label_out(
    input_file1,
    input_file2,
    output_file,
    error_file,
    error_file_original,
    pred_error_ids_path,
):
    error_ids = set()
    correct_data = []
    error_data = []

    with open(input_file1, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get("is_correct") is False:
                error_ids.add(item["id"])
                error_data.append(item)
            else:
                correct_data.append(item)

    error_data_sorted = sorted(error_data, key=lambda x: x["id"])
    with open(error_file, "w", encoding="utf-8") as f:
        json.dump(error_data_sorted, f, indent=4, ensure_ascii=False)
    print(f"Saved data with is_correct=False to {error_file}")

    correct_data_sorted = sorted(correct_data, key=lambda x: x["id"])
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(correct_data_sorted, f, indent=4, ensure_ascii=False)
    print(f"Saved data with is_correct=True to {output_file}")

    with open(pred_error_ids_path, "w", encoding="utf-8") as f:
        for eid in sorted(error_ids):
            f.write(f"{eid}\n")
    print(f"Saved IDs with is_correct=False to {pred_error_ids_path}")

    with open(input_file2, "r", encoding="utf-8") as f:
        data2 = json.load(f)
    filtered_errors = [item for item in data2 if item["id"] in error_ids]
    with open(error_file_original, "w", encoding="utf-8") as f:
        json.dump(filtered_errors, f, indent=4, ensure_ascii=False)

    print(
        f"Saved data corresponding to incorrect IDs from {input_file2} to {error_file_original}"
    )
    os.remove(input_file1)
    print(f"File {input_file1} has been successfully deleted")
    accuracy = round(len(correct_data) / (len(correct_data) + len(error_data)), 2)
    print(f"Accuracy: {accuracy}")


def process_label_in(file1, file2, output_file, final_output_file):

    with open(file1, "r", encoding="utf-8") as f:
        data1 = [json.loads(line.strip()) for line in f]
    data1_sorted = sorted(data1, key=lambda x: x["id"])

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data1_sorted, f, indent=4, ensure_ascii=False)
    print(f"File {file1} has been converted to JSON array and saved to {output_file}")

    with open(file2, "r", encoding="utf-8") as f:
        data2 = json.load(f)
    merged_data = data1_sorted + data2
    sorted_data = sorted(merged_data, key=lambda x: x["id"])

    with open(final_output_file, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, indent=4, ensure_ascii=False)
    print(f"File {final_output_file} has been successfully saved.")

    os.remove(file1)
    print(f"File {file1} has been successfully deleted.")
