/*
 * Verifies document-wide generation invalidation for nested mutations.
 * A reference captured from the root must become invalid when any child
 * object changes, even when the referenced value itself is untouched.
 * This protects callers from retaining pointers across buffer mutations.
 */

#include <assert.h>
#include <stddef.h>

#include "lite3.h"


int main(void)
{
	unsigned char __attribute__((aligned(LITE3_NODE_ALIGNMENT))) buf[1024];
	size_t buflen = 0;
	size_t nested_ofs;
	lite3_str value;

	assert(lite3_init_obj(buf, &buflen, sizeof(buf)) == 0);
	assert(lite3_set_str(buf, &buflen, 0, sizeof(buf), "value", "stable") == 0);
	assert(lite3_set_obj(buf, &buflen, 0, sizeof(buf), "nested", &nested_ofs) == 0);
	assert(lite3_get_str(buf, buflen, 0, "value", &value) == 0);
	assert(LITE3_STR(buf, value) != NULL);

	assert(lite3_set_null(buf, &buflen, nested_ofs, sizeof(buf), "mutated") == 0);
	assert(LITE3_STR(buf, value) == NULL);

	return 0;
}
