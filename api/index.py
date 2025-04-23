from flask import Flask, request, jsonify
from gradio_client import Client, file
import requests
import os
import traceback

app = Flask(__name__)
client = Client("AIRI-Institute/HairFastGAN")
TMPFILES_UPLOAD_URL = "https://tmpfiles.org/api/v1/upload"


def upload_to_tmpfiles(local_path):
    """
    Uploads a local file to tmpfiles.org and returns the direct-download URL.
    """
    with open(local_path, 'rb') as f:
        files = {'file': (os.path.basename(local_path), f)}
        resp = requests.post(TMPFILES_UPLOAD_URL, files=files)

    if resp.status_code != 200:
        raise Exception(f"Upload failed: {resp.status_code}: {resp.text}")

    try:
        result = resp.json()
        # handle nested data structure
        url = result.get('data', {}).get('url') or result.get('url') or result.get('link') or result
    except ValueError:
        url = resp.text.strip()

    # convert to direct-download link
    if isinstance(url, str) and url.startswith("https://tmpfiles.org/") and "/dl/" not in url:
        url = url.replace("https://tmpfiles.org/", "https://tmpfiles.org/dl/")

    return url


@app.route('/process-hair-swap', methods=['POST'])
def process_hair_swap():
    data = request.get_json(force=True)

    face_url = data.get('face_url')
    shape_url = data.get('shape_url')
    color_url = data.get('color_url')

    if not all([face_url, shape_url, color_url]):
        return jsonify({"error": "face_url, shape_url, and color_url are required"}), 400

    local_files = []
    try:
        # 1) Resize face
        face_local = client.predict(
            img=file(face_url),
            align=["Face"],
            api_name="/resize_inner"
        )
        local_files.append(face_local)
        face_dl = upload_to_tmpfiles(face_local)

        # 2) Resize shape
        shape_local = client.predict(
            img=file(shape_url),
            align=["Shape"],
            api_name="/resize_inner_1"
        )
        local_files.append(shape_local)
        shape_dl = upload_to_tmpfiles(shape_local)

        # 3) Resize color
        color_local = client.predict(
            img=file(color_url),
            align=["Color"],
            api_name="/resize_inner_2"
        )
        local_files.append(color_local)
        color_dl = upload_to_tmpfiles(color_local)

        # 4) Swap hair
        swap_output = client.predict(
            face=file(face_dl),
            shape=file(shape_dl),
            color=file(color_dl),
            blending=data.get('blending', "Article"),
            poisson_iters=int(data.get('poisson_iters', 2500)),
            poisson_erosion=int(data.get('poisson_erosion', 100)),
            api_name="/swap_hair"
        )
        # handle tuple response for swap_hair
        if isinstance(swap_output, (tuple, list)):
            # extract the first visible image path
            swapped_local = None
            for item in swap_output:
                if isinstance(item, dict) and item.get('visible') and 'value' in item:
                    swapped_local = item['value']
                    break
            if not swapped_local:
                raise Exception(f"Unexpected swap output format: {swap_output}")
        else:
            swapped_local = swap_output

        local_files.append(swapped_local)

        # 5) Upload final swapped image
        swapped_dl = upload_to_tmpfiles(swapped_local)
        return jsonify({"result_url": swapped_dl}), 200

    except Exception as e:
        print("Exception occurred:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        # clean up temp files
        for path in local_files:
            try:
                if path and os.path.isfile(path):
                    os.remove(path)
            except Exception:
                pass  # ignore cleanup failures


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
