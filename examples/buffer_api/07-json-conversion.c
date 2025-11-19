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
#include <stdlib.h>

#include "lite3.h"


static unsigned char buf[256 * 1024];

int main() {
        size_t buflen = 0;
        size_t bufsz = sizeof(buf);

        // Convert JSON file to Lite³
        if (lite3_json_dec_file(buf, &buflen, bufsz, "examples/periodic_table.json") < 0) {
                perror("Failed to decode JSON document");
                return 1;
        }

        // Iterator to find densest element
        size_t data_ofs;
        if (lite3_get_arr(buf, buflen, 0, "data", &data_ofs) < 0) {
                perror("Failed to get data array");
                return 1;
        }
        lite3_iter iter;
        if (lite3_iter_create(buf, buflen, data_ofs, &iter) < 0) {
                perror("Failed to create iterator");
                return 1;
        }
        size_t el_ofs;
        size_t el_densest_ofs = 0;
        double el_densest_kg_per_m3 = 0.0;
        int ret;
        while ((ret = lite3_iter_next(buf, buflen, &iter, NULL, &el_ofs)) == LITE3_ITER_ITEM) {
                if (lite3_is_null(buf, buflen, el_ofs, "density_kg_per_m3")) {
                        continue;
                }
                double kg_per_m3;
                if (lite3_get_f64(buf, buflen, el_ofs, "density_kg_per_m3", &kg_per_m3) < 0) {
                        perror("Failed to get element density");
                        return 1;
                }
                if (kg_per_m3 > el_densest_kg_per_m3) {
                        el_densest_ofs = el_ofs;
                        el_densest_kg_per_m3 = kg_per_m3;
                }
        }
        if (ret < 0) {
                perror("Failed to get iter element");
                return 1;
        }
        if (el_densest_ofs == 0) {
                perror("Failed to find densest element");
                return 1;
        }

        lite3_str name;
        if (lite3_get_str(buf, buflen, el_densest_ofs, "name", &name) < 0) {
                perror("Failed to get densest element name");
                return 1;
        }
        printf("densest element: %s\n\n", LITE3_STR(buf, name));

        printf("Convert Lite³ to JSON by returned heap pointer (prettified):\n");
        size_t json_len;
        char *json = lite3_json_enc_pretty(buf, buflen, el_densest_ofs, &json_len);
        if (!json) {
                perror("Failed encode JSON");
                return 1;
        }
        printf("%s\n\n", json);
        free(json);

        printf("Convert Lite³ to JSON by writing to buffer (non-prettified):\n");
        size_t json_buf_size = 1024;
        char *json_buf = malloc(json_buf_size);
        int64_t ret_i64;
        if ((ret_i64 = lite3_json_enc_buf(buf, buflen, el_densest_ofs, json_buf, json_buf_size)) < 0) {
                perror("Failed encode JSON");
                return 1;
        }
        size_t json_buf_len = (size_t)ret_i64;
        printf("%s\n", json_buf);
        printf("json bytes written: %zu\n", json_buf_len);
        free(json_buf);

        return 0;
}