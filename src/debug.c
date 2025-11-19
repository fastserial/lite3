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
#include "lite3.h"



#ifdef LITE3_DEBUG
#include <stdio.h>



void lite3_print(const unsigned char *buf, size_t buflen)
{
        for (size_t i = 0; i < buflen; i++) {
                unsigned char c = buf[i];
                if(c >= 0x20 && c <= 0x7E) {
                        putchar(c);
                        putchar(' ');
                } else {
                        putchar("0123456789ABCDEF"[c >> 4]);
                        putchar("0123456789ABCDEF"[c & 0xF]);
                }

                if (!((i + 1) & 3)) {
                        if (!((i + 1) & 63u)) {
                                printf("\t%zu\n\n", i + 1);
                        } else if (!((i + 1) & 31)) {
                                putchar('\n');

                        } else {
                                putchar(' ');
                        }
                }
        }
        putchar('\n');
}
#endif // LITE3_DEBUG