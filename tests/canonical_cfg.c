#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "lite3.h"

static void assert_cfg_id(const unsigned char *buf, size_t buflen, size_t ofs, enum lite3_node_cfg_id expected)
{
        const lite3_node_cfg *cfg = NULL;
        if (_lite3_cfg_for_offset(buf, buflen, ofs, &cfg, NULL) < 0) {
                perror("Failed to read node config");
                assert(0);
        }
        assert(cfg && cfg->id == expected);
}

int main(void)
{
        unsigned char buf[4096] __attribute__((aligned(LITE3_NODE_ALIGNMENT)));
        size_t buflen = 0;

        const char *json_small_obj = "{\"a\":1,\"b\":2}";
        if (lite3_json_dec(buf, &buflen, sizeof(buf), json_small_obj, strlen(json_small_obj)) < 0) {
                perror("Failed to decode JSON (small obj)");
                return 1;
        }
        assert_cfg_id(buf, buflen, 0, LITE3_NODE_CFG_48);

        const char *json_arr_8 = "[0,1,2,3,4,5,6,7]";
        if (lite3_json_dec(buf, &buflen, sizeof(buf), json_arr_8, strlen(json_arr_8)) < 0) {
                perror("Failed to decode JSON (array)");
                return 1;
        }
        assert_cfg_id(buf, buflen, 0, LITE3_NODE_CFG_192);

        const char *json_nested =
                "{\"small\":{\"a\":1},"
                "\"big\":{\"k1\":1,\"k2\":2,\"k3\":3,\"k4\":4,\"k5\":5,\"k6\":6,\"k7\":7,\"k8\":8},"
                "\"arr\":[1,2,3,4]}";
        if (lite3_json_dec(buf, &buflen, sizeof(buf), json_nested, strlen(json_nested)) < 0) {
                perror("Failed to decode JSON (nested)");
                return 1;
        }
        assert_cfg_id(buf, buflen, 0, LITE3_NODE_CFG_48);

        size_t small_ofs = 0;
        size_t big_ofs = 0;
        size_t arr_ofs = 0;
        if (lite3_get_obj(buf, buflen, 0, "small", &small_ofs) < 0) {
                perror("Failed to read small object");
                return 1;
        }
        if (lite3_get_obj(buf, buflen, 0, "big", &big_ofs) < 0) {
                perror("Failed to read big object");
                return 1;
        }
        if (lite3_get_arr(buf, buflen, 0, "arr", &arr_ofs) < 0) {
                perror("Failed to read array");
                return 1;
        }

        assert_cfg_id(buf, buflen, small_ofs, LITE3_NODE_CFG_48);
        assert_cfg_id(buf, buflen, big_ofs, LITE3_NODE_CFG_192);
        assert_cfg_id(buf, buflen, arr_ofs, LITE3_NODE_CFG_96);

        return 0;
}
