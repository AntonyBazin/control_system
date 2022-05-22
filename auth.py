import requests


class Auth:
    """Class to make authenticated requests."""

    def __init__(self, access_token: str):
        """Initialize the auth."""
        self.access_token = access_token

    def request(self, method: str, path: str, **kwargs) -> requests.Response:
        """Make a request."""
        headers = kwargs.pop('headers')

        if headers is None:
            headers = {}
        else:
            headers = dict(headers)

        headers['authorization'] = 'Bearer ' + self.access_token

        return requests.request(
            method, f'{path}', **kwargs, headers=headers,)
