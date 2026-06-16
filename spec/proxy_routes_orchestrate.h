#pragma once
#include "httplib.h"

/** Register POST /api/orchestrate and POST /api/orchestrate/stream on the proxy server. */
void register_proxy_orchestrate_routes(httplib::Server& svr);
