import os
import sys
import json
import slack_sdk
import requests


def load_secret(name):
    with open("slack.json", "r") as f:
        secret = json.load(f)[name]
    return secret

def make_slack_format(text: str):
    return json.dumps({"text": text})

class SlackMessenger:
    def __init__(self):
        name = "Slack"
        secret = load_secret(name)
        self.channel = secret["CHANNEL"]
        self.token = secret["ACCESSED_TOKEN"]
        self.web_hook_url = secret["WEB_HOOK_URL"]
        self.client = slack_sdk.WebClient(token=self.token)

    def send_file(self, file_path, file_title):
        response = self.client.files_upload(
            channels=self.channel,
            file=file_path,
            title=file_title,
            filetype='png'
        )

    def alarm_msg(self, title, alarm_text, colour="9999FF"):
        slack_text = make_alarm_format(title, alarm_text, colour)
        response = requests.post(self.web_hook_url, data=slack_text, headers={'Content-Type': 'application/json'})
        if response.status_code != 200:
            raise ValueError(response.status_code, response.text)

def make_alarm_format(title: str, text: str, colour):
        result = {"attachments": [
            {
                "color": colour,
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "plain_text",
                            "text": f"{title}"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "plain_text",
                                "text": f"{text}"
                            }
                        ]
                    }
                ]}]}
        return json.dumps(result)
