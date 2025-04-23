from flask import Flask, request, jsonify
from gradio_client import Client, file
import requests
import os
import traceback

app = Flask(__name__)
client = Client("AIRI-Institute/HairFastGAN")
TMPFILES_UPLOAD_URL = "https://tmpfiles.org/api/v1/upload"


def upload_to_tmpfiles(local_path):
    with open(local_path, 'rb') as f:
        files = {'file': (os.path.basename(local_path), f)}
        resp = requests.post(TMPFILES_UPLOAD_URL, files=files)
    if resp.status_code != 200:
        raise Exception(f"Upload failed: {resp.text}")
    try:
        result = resp.json()
        url = result['data']['url'] if isinstance(result, dict) else result
    except Exception:
        url = resp.text.strip()
    return url.replace("https://tmpfiles.org/", "https://tmpfiles.org/dl/")


@app.route('/process-hair-swap', methods=['POST'])
def process_hair_swap():
    data = request.get_json()

    face_url = data.get("face_url")
    shape_url = data.get("shape_url")
    color_url = data.get("color_url")

    if not face_url or not shape_url or not color_url:
        return jsonify({"error": "Missing one or more image URLs"}), 400

    try:
        # Resize step for face
        face_img = client.predict(
            img=file(face_url),
            align=["Face", "Shape", "Color"],
            api_name="/resize_inner"
        )
        face_dl_url = upload_to_tmpfiles(face_img)

        # Resize step for shape
        shape_img = client.predict(
            img=file(shape_url),
            align=["Face", "Shape", "Color"],
            api_name="/resize_inner_1"
        )
        shape_dl_url = upload_to_tmpfiles(shape_img)

        # Resize step for color
        color_img = client.predict(
            img=file(color_url),
            align=["Face", "Shape", "Color"],
            api_name="/resize_inner_2"
        )
        color_dl_url = upload_to_tmpfiles(color_img)

        # Hair swap
        result_img = client.predict(
            face=file(face_dl_url),
            shape=file(shape_dl_url),
            color=file(color_dl_url),
            blending=data.get("blending", "Article"),
            poisson_iters=int(data.get("poisson_iters", 2500)),
            poisson_erosion=int(data.get("poisson_erosion", 100)),
            api_name="/swap_hair"
        )

        result_dl_url = upload_to_tmpfiles(result_img)

        return jsonify({
            "result_url": result_dl_url
        })

    except Exception as e:
        print("Exception occurred:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
