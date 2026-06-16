#include "proxy_routes_orchestrate.h"
#include "matrix_env.h"
#include "json.hpp"
#include <iostream>

using json = nlohmann::json;

void register_proxy_orchestrate_routes(httplib::Server& svr) {
    auto cors = [](httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
    };

    // SSE streaming passthrough for Python orchestration modes.
    // Pre-flight validates mode + prompt before committing to 200 + chunked.
    svr.Post("/api/orchestrate/stream", [cors](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");
        std::string req_body = req.body;
        try {
            auto body = json::parse(req_body);
            if (body.value("mode", "").empty()) {
                res.status = 400;
                res.set_content(json{{"error", "'mode' required"}}.dump(), "application/json");
                return;
            }
            if (body.value("prompt", "").empty()) {
                res.status = 400;
                res.set_content(json{{"error", "'prompt' required"}}.dump(), "application/json");
                return;
            }
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            return;
        }
        int port = g_env.python_coord_port;
        res.set_chunked_content_provider("text/event-stream",
            [req_body, port](size_t, httplib::DataSink& sink) {
                httplib::Client py("127.0.0.1", port);
                py.set_connection_timeout(5);
                py.set_read_timeout(300);
                httplib::Headers hdrs;
                py.Post("/api/orchestrate/stream", hdrs,
                    req_body.size(),
                    [&req_body](size_t off, size_t len, httplib::DataSink& ds) {
                        size_t n = std::min(len, req_body.size() > off
                            ? req_body.size() - off : (size_t)0);
                        if (n > 0) ds.write(req_body.data() + off, n);
                        return true;
                    },
                    "application/json",
                    [&sink](const char* data, size_t len) {
                        return sink.write(data, len);
                    });
                sink.done();
                return false;
            });
    });

    // Blocking JSON passthrough — returns full result once Python coord completes.
    svr.Post("/api/orchestrate", [cors](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        try {
            auto body = json::parse(req.body);
            if (body.value("mode", "").empty()) {
                res.status = 400;
                res.set_content(json{{"error", "'mode' required"}}.dump(), "application/json");
                return;
            }
            httplib::Client py_coord("127.0.0.1", g_env.python_coord_port);
            py_coord.set_connection_timeout(5);
            py_coord.set_read_timeout(300);
            auto r = py_coord.Post("/api/orchestrate", req.body, "application/json");
            if (r) {
                res.status = r->status;
                std::string ct = r->get_header_value("Content-Type");
                res.set_content(r->body, ct.empty() ? "application/json" : ct.c_str());
            } else {
                res.status = 503;
                res.set_content(
                    json{{"error", "Python coordinator offline — run: brewctl launch"}}.dump(),
                    "application/json");
            }
        } catch (const std::exception& e) {
            std::cerr << "[/api/orchestrate] " << e.what() << "\n";
            res.status = 500;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });
}
