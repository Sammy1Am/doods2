import odrpc
import json
import base64
import logging
import asyncio
import threading
import paho.mqtt.client as mqtt
import cv2
import numpy as np
from streamer import Streamer

class MQTT():
    def __init__(self, config, doods):
        self.config = config
        self.doods = doods
        self.mqtt_client = mqtt.Client()
        # Borrow the uvicorn logger because it's pretty.
        self.logger = logging.getLogger("doods.api")

    async def stream(self, detect_request: str = '{}'):
        streamer = None
        try:
            # Run the stream detector and return the results.
            streamer = Streamer(self.doods).start_stream(detect_request)
            for detect_response in streamer:
                # If we requested an image get it from the response
                mqtt_image = None
                if detect_request.image:
                    mqtt_image = detect_response.image
                    detect_image = None
                    # If we need to crop the image, decode it to prepare for cropping (outside detection loop
                    # so multiple detections can use the same decoded image)
                    if detect_request.crop:
                        detect_image_bytes = np.frombuffer(detect_response.image, dtype=np.uint8)
                        detect_image = cv2.imdecode(detect_image_bytes, cv2.IMREAD_COLOR)
                        di_height, di_width = detect_image.shape[:2]
                for detection in detect_response.detections:
                    self.mqtt_client.publish(
                        f"doods/detect/{detect_request.id}{'' if detection.region_id is None else '/'+detection.region_id}", 
                        payload=json.dumps(detection.asdict(include_none=False)), qos=0, retain=False)
                    # If we requested a cropped image, do cropping
                    if detect_image is not None:
                        cropped_image = detect_image[
                            int(detection.top*di_height):int(detection.bottom*di_height), 
                            int(detection.left*di_width):int(detection.right*di_width)]
                        mqtt_image = cv2.imencode(detect_request.image, cropped_image)[1].tostring()
                    # If we requested an image, publish to MQTT
                    if mqtt_image is not None:
                        self.mqtt_client.publish(
                        f"doods/image/{detect_request.id}{'' if detection.region_id is None else '/'+detection.region_id}/{detection.label or 'object'}", 
                        payload=mqtt_image, qos=0, retain=False)

        except Exception as e:
            self.logger.info(e)
            try:
                if streamer:
                    streamer.send(True)  # Stop the streamer
            except StopIteration:
                pass

    def run(self):
        if (self.config.broker.user):
            self.mqtt_client.username_pw_set(self.config.broker.user, self.config.broker.password)
        self.mqtt_client.connect(self.config.broker.url, self.config.broker.port, 60)
        for request in self.config.requests:
            asyncio.run(self.stream(request))
