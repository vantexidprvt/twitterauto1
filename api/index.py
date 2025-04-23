from flask import Flask, request, jsonify
from gradio_client import Client, file
import requests
import os

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

    # Parse JSON or fallback to raw text
    try:
        result = resp.json()
        url = (result.get('data', {}).get('url') or
               result.get('url') or
               result.get('link') or
               result)
    except ValueError:
        url = resp.text.strip()

    # Ensure direct-download link
    if isinstance(url, str) and url.startswith("https://tmpfiles.org/") and "/dl/" not in url:
        url = url.replace("https://tmpfiles.org/", "https://tmpfiles.org/dl/")

    return url

@app.route('/process-hair-swap', methods=['POST'])
def process_hair_swap():
    data = request.get_json()
    face_url = data.get('face_url')
    shape_url = data.get('shape_url')
    color_url = data.get('color_url')

    # Validate inputs
    if not all([face_url, shape_url, color_url]):
        return jsonify({"error": "face_url, shape_url, and color_url are required"}), 400

    local_files = []
    try:
        # 1) Resize face
        face_resized = client.predict(
            img=file(face_url),
            align=["Face"],
            api_name="/resize_inner"
        )
        local_files.append(face_resized)
        face_dl = upload_to_tmpfiles(face_resized)

        # 2) Resize shape
        shape_resized = client.predict(
            img=file(shape_url),
            align=["Shape"],
            api_name="/resize_inner_1"
        )
        local_files.append(shape_resized)
        shape_dl = upload_to_tmpfiles(shape_resized)

        # 3) Resize color
        color_resized = client.predict(
            img=file(color_url),
            align=["Color"],
            api_name="/resize_inner_2"
        )
        local_files.append(color_resized)
        color_dl = upload_to_tmpfiles(color_resized)

        # 4) Swap hair
        swapped = client.predict(
            face=file(face_dl),
            shape=file(shape_dl),
            color=file(color_dl),
            blending=data.get('blending', "Article"),
            poisson_iters=data.get('poisson_iters', 2500),
            poisson_erosion=data.get('poisson_erosion', 100),
            api_name="/swap_hair"
        )
        local_files.append(swapped)

        # 5) Upload final image
        swapped_dl = upload_to_tmpfiles(swapped)
        return jsonify({"result_url": swapped_dl}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temporary files
        for path in local_files:
            if path and os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

if __name__ == '__main__':
    app.run(debug=True)
