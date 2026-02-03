import base64


def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_string = base64.b64encode(img_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_string}"


def image_to_base64(file_path):
    with open(file_path, "rb") as img_file:
        base64_string = base64.b64encode(img_file.read()).decode("utf-8")
    return base64_string


