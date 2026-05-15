import ssl, urllib.request, certifi

cert_file = "/Users/yanbin.qiu/Downloads/ZscalerRootCA-Feb2025.pem"

_ssl_ctx = ssl.create_default_context(cafile=certifi.where())
_ssl_ctx.load_verify_locations(cert_file)
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ssl_ctx))
)

import os
os.environ["SSL_CERT_FILE"] = cert_file
os.environ["REQUESTS_CA_BUNDLE"] = cert_file

print("SSL certificate configured successfully.")
