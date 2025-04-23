from flask import Flask, request, jsonify
from gradio_client import Client, file
import requests
import tempfile
import threading
import os
import mimetypes

app = Flask(__name__)
client = Client("AIRI-Institute/HairFastGAN")

UPLOAD_API = "https://tmpfiles.org/api/v1/upload"

# Function to download and upload the image to tmpfiles.org
def download_and_upload(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()

        # Get extension from URL or content-type header
        ext = os.path.splitext(image_url)[1]
        if not ext:
            ext = mimetypes.guess_extension(response.headers.get("Content-Type", "image/webp"))
        ext = ext if ext else ".webp"

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file.flush()
            with open(temp_file.name, "rb") as f:
                uploaded = requests.post(UPLOAD_API, files={"file": f})
            uploaded.raise_for_status()
            url_part = uploaded.text.split("/")[-1].strip()
            return f"https://tmpfiles.org/dl/{url_part}"
    except Exception as e:
        raise Exception(f"Error processing {image_url}: {e}")

@app.route('/process', methods=['POST'])
def process_images():
    data = request.json
    face_url = data.get("face")
    shape_url = data.get("shape")
    color_url = data.get("color")

    if not all([face_url, shape_url, color_url]):
        return jsonify({"error": "Missing required image URLs (face, shape, color)"}), 400

    result = {}

    def call_resize(endpoint, url, key):
        try:
            # Step 1: Download the image and upload it to tmpfiles.org
            link = download_and_upload(url)
            result[key] = {"link": link, "processed": client.predict(img=file(link), align=["Face", "Shape", "Color"], api_name=endpoint)}
        except Exception as e:
            result[key] = {"error": str(e)}

    # Step 2: Process each image (face, shape, color) in parallel
    threads = [
        threading.Thread(target=call_resize, args=("/resize_inner", face_url, "face")),
        threading.Thread(target=call_resize, args=("/resize_inner_1", shape_url, "shape")),
        threading.Thread(target=call_resize, args=("/resize_inner_2", color_url, "color")),
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if any("error" in result[k] for k in ["face", "shape", "color"]):
        return jsonify({"error": "One or more resize stages failed", "details": result}), 500

    try:
        # Step 3: Perform the final swap
        final_image = client.predict(
            face=file(result["face"]["processed"]),
            shape=file(result["shape"]["processed"]),
            color=file(result["color"]["processed"]),
            blending="Article",
            poisson_iters=2500,
            poisson_erosion=100,
            api_name="/swap_hair"
        )

        # Step 4: Upload the final result to tmpfiles.org
        with open(final_image, "rb") as final_img:
            final_upload = requests.post(UPLOAD_API, files={"file": final_img})
            final_upload.raise_for_status()
            url_part = final_upload.text.split("/")[-1].strip()
            final_link = f"https://tmpfiles.org/dl/{url_part}/image.webp"
            
            # Step 5: Return the processed resized images and final result link
            return jsonify({
                "resized": {
                    "face": result["face"]["link"],
                    "shape": result["shape"]["link"],
                    "color": result["color"]["link"]
                },
                "result_url": final_link
            })

    except Exception as e:
        return jsonify({"error": "Final hair swap failed", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
