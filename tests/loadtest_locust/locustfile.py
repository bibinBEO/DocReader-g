
from locust import HttpUser, task, between
import os

class DocReaderUser(HttpUser):
    wait_time = between(1, 2) # Users wait between 1 and 2 seconds between tasks

    host = os.getenv("LOCUST_HOST", "http://localhost:8000")
    auth_token = os.getenv("LOCUST_AUTH_TOKEN", "Bearer your_jwt_token_here")

    def on_start(self):
        self.client.headers = {"Authorization": self.auth_token}

    @task(3)
    def upload_document(self):
        # Assuming you have a dummy PDF file for testing
        # You might need to create one in your tests/loadtest_locust directory
        file_path = "./tests/loadtest_locust/dummy.pdf"
        if not os.path.exists(file_path):
            print(f"WARNING: Dummy PDF not found at {file_path}. Please create one for load testing.")
            # Create a very basic dummy PDF if it doesn't exist
            try:
                from reportlab.pdfgen import canvas
                c = canvas.Canvas(file_path)
                c.drawString(100, 750, "This is a dummy PDF for load testing.")
                c.save()
                print(f"Created dummy PDF at {file_path}")
            except ImportError:
                print("ERROR: reportlab not installed. Cannot create dummy PDF. Please install it (pip install reportlab).")
                return

        with open(file_path, "rb") as f:
            response = self.client.post("/upload", files={"file": f})
            if response.status_code == 200:
                task_id = response.json().get("task_id")
                if task_id:
                    self.task_id = task_id # Store task_id for status check
                    self.document_id = response.json().get("document_id")
                else:
                    print(f"Upload successful but no task_id in response: {response.json()}")
            else:
                print(f"Upload failed with status {response.status_code}: {response.text}")

    @task(1)
    def check_status(self):
        if hasattr(self, 'task_id'):
            self.client.get(f"/status/{self.task_id}")
        else:
            print("No task_id available to check status.")

    # You might add a task to check results, but it depends on how quickly
    # the Celery worker processes the documents.
    # @task(1)
    # def get_result(self):
    #     if hasattr(self, 'document_id'):
    #         self.client.get(f"/result/{self.document_id}")
    #     else:
    #         print("No document_id available to get result.")
