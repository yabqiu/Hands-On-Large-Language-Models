import ssl, urllib.request, certifi

_ssl_ctx = ssl.create_default_context(cafile=certifi.where())
_ssl_ctx.load_verify_locations("/Users/yanbin.qiu/Downloads/ZscalerRootCA-Feb2025.pem")
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ssl_ctx))
)