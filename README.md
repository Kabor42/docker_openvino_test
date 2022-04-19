# Test OpenVINO Runtime

Testing runtime environment especially with MYRIAD or NCS2 USB device.

# Build

## Requirements

- Docker

## Build image

Preferably name your image to something findable, like *openvino_docker_test:1.0*.

```bash
docker build . -t <custom_image_name>
```

The build should end within 10 minutes with a considerably good setup.

# Run

The image will run any line in the *Dockerfile* starting with **CMD**.
Please alter if necessary.

```
docker run \
    --it \
    --rm \
    --privileged \
    -v /dev:/dev \
    --network host \
    <custom_image_name>
```

If you forgot your image name, you can find it in the list with `docker images`.

> In order to properly initialize the MYRIAD device we must run it as *privileged*,
> or build a custom *libusb* without *udev* suport.
> Refer to: [OpenVINO Docker Setup](https://docs.openvino.ai/2021.3/openvino_docs_install_guides_installing_openvino_docker_linux.html)

# Modification

You can modify the program all you want, but don't forget to rebuild the image.
