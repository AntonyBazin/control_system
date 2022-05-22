import requests


class Auth:
    """Class to make authenticated requests."""

    def __init__(self, address: str, access_token: str):
        """Initialize the auth."""
        self.address = address
        self.access_token = access_token

    def request(self, method: str, **kwargs) -> requests.Response:
        """Make a request."""
        headers = kwargs.pop('headers')

        if headers is None:
            headers = {}
        else:
            headers = dict(headers)

        headers['authorization'] = 'Bearer ' + self.access_token

        return requests.request(
            method, f'{self.address}', **kwargs, headers=headers,)
