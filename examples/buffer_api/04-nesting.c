/*
    Lite³: A JSON-Compatible Zero-Copy Serialization Format

    Copyright © 2025 Elias de Jong <elias@fastserial.com>

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

      __ __________________        ____
    _  ___ ___/ /___(_)_/ /_______|_  /
     _  _____/ / __/ /_  __/  _ \_/_ < 
      ___ __/ /___/ / / /_ /  __/____/ 
           /_____/_/  \__/ \___/       
*/
#include <stdio.h>
#include <string.h>

#include "lite3.h"


static unsigned char buf[1024];

int main() {
        size_t buflen = 0;
        size_t bufsz = sizeof(buf);

        // Build message
        if (lite3_init_obj(buf, &buflen, bufsz)                                     < 0
                || lite3_set_str(buf, &buflen, 0, bufsz, "event", "http_request")   < 0
                || lite3_set_str(buf, &buflen, 0, bufsz, "method", "POST")          < 0
                || lite3_set_i64(buf, &buflen, 0, bufsz, "duration_ms", 47)         < 0) {
                perror("Failed to build message");
                return 1;
        }
        // Set headers
        size_t headers_ofs;
        if (lite3_set_obj(buf, &buflen, 0, bufsz, "headers", &headers_ofs)                              < 0
                || lite3_set_str(buf, &buflen, headers_ofs, bufsz, "content-type", "application/json")  < 0
                || lite3_set_str(buf, &buflen, headers_ofs, bufsz, "x-request-id", "req_9f8e2a")        < 0
                || lite3_set_str(buf, &buflen, headers_ofs, bufsz, "user-agent", "curl/8.1.2")          < 0) {
                perror("Failed to set headers");
                return 1;
        }

        if (lite3_json_print(buf, buflen, 0) < 0) { // Print Lite³ as JSON
                perror("Failed to print JSON");
                return 1;
        }

        // Get user-agent
        lite3_str user_agent;
        size_t ofs;
        if (lite3_get_obj(buf, buflen, 0, "headers", &ofs)                      < 0
                || lite3_get_str(buf, buflen, ofs, "user-agent", &user_agent)   < 0) {
                perror("Failed to get user-agent");
                return 1;
        }
        printf("User agent: %s\n", LITE3_STR(buf, user_agent));

        return 0;
}