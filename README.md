# Lite³: A JSON-Compatible Zero-Copy Serialization Format
*Parse no more—the wire format is the memory format.*

![](img/lite3_seamless_dark.png)
![](img/lite3_infographic_dark.png)

<h2><a href="https://lite3.io/topics.html">Official Documentation (with examples): lite3.io</a></h2>

## Introduction
Lite³ is a JSON-compatible zero-copy serialization format able to encode semi-structured data in a lightweight binary format, suitable for embedded and no-malloc environments. The flagship feature is the possibility to apply **mutations directly on the serialized form**. With Lite³, you can insert any arbitrary key, with any arbitrary value, directly into a serialized message. Essentially, it functions as a *serialized dictionary*.  
Some other formats provide this, but only limited to in-place updates of fixed size. As of writing (Nov 15th, 2025), this is a capability not provided by any other format.
Additionally, Lite³ implements this **without additional memory allocations** and always guaranteeing `O(log n)` amortized time complexity with predicatable latency for IOPS.

As a result, the serialization boundary has been broken: 'parsing' or 'serializing' in the traditional sense is no longer necessary. Lite³ structures can be read and mutated similar to hashmaps or binary trees, and since they exist in a single contiguous buffer, they always remain ready to send.
Other state-of-the-art binary- and zero-copy formats still require full message reserialization for any non-trivial mutation.

Compared to other binary formats, Lite³ is also schemaless, self-describing (no IDL or schema definitons required) and **fully compatible with JSON**, enabling seamless conversion between the two formats. This ensures compatibility with many existing datasets and APIs while also allowing for easy debugging/inspecting of messages.

Example to illustrate:
1. A Lite³ message is received from a socket
2. Without doing any parsing, the user can immediately:
        - Lookup keys and read values via zero-copy pointers
        - Insert/overwrite arbitrary key/value entries
3. After all operations are done, the structure can be transmitted 'as-is' (no serialization required, just `memcpy()`)
4. The receiver then has access to all the same operations

Lite³ blurs the line between memory and wire formats. It eliminates several steps typically required in computer communications, unlocking new potential for realtime, embedded and high-performance applications.


## Features
- Schemaless & self-describing, no IDL or schema definitons required
- Zero-copy reads + writes of any data size
- Lives on OSI layer 6 (transport/protocol agnostic)
- O(log n) amortized time complexity for all IOPS
- Built-in pointer validation
- Low memory profile
- Predictable latency
- No `malloc()` API, caller provides buffer
- Library size 9.3 kB (core) and dependency-free
- Fully written in gcc/clang C11
- Optional subdependency (yyjson) to support conversion to/from JSON
- MIT license


### Feature Matrix

| Format name                   | Schemaless    | Zero-copy reads[^1]   | Zero-copy writes[^2]  | Human-readable[^3]       |
| ----------------------------- | ------------- | --------------------- | --------------------- | ------------------------ |
| Lite³                         | ✅            | ✅ O(log n)          | ✅ O(log n)           | ⚠️ (convertable to JSON)  |
| JSON                          | ✅            | ❌                   | ❌                    | ✅                        |
| BSON                          | ✅            | ❌                   | ❌                    | ⚠️ (convertable to JSON)  |
| MessagePack                   | ✅            | ❌                   | ❌                    | ⚠️ (convertable to JSON)  |
| CBOR                          | ✅            | ❌                   | ❌                    | ⚠️ (convertable to JSON)  |
| Smile                         | ✅            | ❌                   | ❌                    | ⚠️ (convertable to JSON)  |
| Ion (Amazon)                  | ✅            | ❌                   | ❌                    | ⚠️ (convertable to JSON)  |
| Protobuf (Google)             | ❌            | ❌                   | ❌                    | ❌[^4]                    |
| Apache Arrow (based on Flatb.)| ❌            | ✅ O(1)              | ❌ (immutable)        | ❌                        |
| Flatbuffers (Google)          | ❌            | ✅ O(1)              | ❌ (immutable)        | ❌                        |
| Flexbuffers (Google)          | ✅            | ✅[^5]               | ❌ (immutable)        | ⚠️ (convertable to JSON)  |
| Cap'n Proto (Cloudflare)      | ❌            | ✅ O(1)              | ⚠️ (in-place only)    | ❌                        |
| Thrift (Facebook)             | ❌            | ❌                   | ❌                    | ❌                        |
| Avro (Apache)                 | ❌            | ❌                   | ❌                    | ❌                        |
| Bond (Microsoft, discontinued)| ❌            | ⚠️ (limited)         | ❌                    | ❌                        |
| DER (ASN.1)                   | ❌            | ⚠️ (limited)         | ❌                    | ❌                        |
| SBE                           | ❌            | ✅ O(1)              | ⚠️ (in-place only)    | ❌                        |

[^1]: Zero-copy reads: The ability to perform arbitrary lookups inside the structure without deserializing or parsing it first.
[^2]: Zero-copy writes: The ability to perform arbitrary mutations inside the structure without deserializing or parsing it first.
[^3]: To be considered human-readable, all necessary information must be provided in-band (no outside schema).
[^4]: Protobuf can optionally send messages in 'ProtoJSON' format for debugging, but in production systems they are still sent as binary and not inspectable without schema. Other binary formats also support similar features, however we do not consider these formats 'human-readable' since they rely on out-of-band information.
[^5]: Flexbuffer access to scalars and vectors is `O(1)` (ints, floats, etc.). For maps, access is `O(log n)`.

Remember that we judge the behavior of formats by their implementation rather than by their official spec. This is because we cannot judge the behavior of hypothetical non-existant implementations.


## Benchmarks
### Simdjson Twitter API Data Benchmark
This benchmark by the authors of [the official simdjson respository](https://github.com/simdjson/simdjson)
was created to compare JSON parsing performance for different C/C++ libraries.

An input dataset `twitter.json` is used, consisting ~632 kB of real twitter API data to perform a number of tasks, each having its own category:
1. **top_tweet**: Find the tweet with the most number of retweets.
2. **partial_tweets**: Iterate over all tweets, extracting only a number of fields and storing it inside an `std::vector`.
3. **find_tweet**: Find a tweet inside the dataset with a specific ID.
4. **distinct_user_id**: Collect all unique user IDs inside the dataset and store it inside an `std::vector<uint64_t>`.

While these tasks are intended to compare JSON parsing performance, they represent real patterns inside applications in which data might be queried.

Text formats do not contain enough information for a parser to know the structure of the document immediately.
This structure must be 'discovered' by finding brackets, commas, semicolons etc.
Through this process, the parser acquires information necessary for traversal.
An unfortunate result of this, is that typically the entire dataset must be fed through the CPU, even if a query is only interested in a subset or single field.

A zero-copy format will approach each problem in a different way.
It already contains all the information necessary to find internal fields.
Only some index structure is required, along with fields of interest. The rest of the dataset is irrelevant to the CPU and might never even enter cache.
Therefore to answer a query like 'find tweet by ID', the actual bytes read may be counted only in the hundreds or low thousands out of ~632 kB.

Converting the dataset to Lite³ (a zero-copy format) to answer the exact same queries
presents an opportunity to quantify this advantage and reveal something about the cost of text formats.

![](img/lite3_benchmark_simdjson_twitter_api_data.png)

| Format               | top_tweet        | partial_tweets   | find_tweet       | distinct_user_id |
| -------------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| yyjson               | 205426 ns        | -                | 203147 ns        | 207233 ns        |
| simdjson On-Demand   | 91184 ns         | 91090 ns         | 53937 ns         | 85036 ns         |
| simdjson DOM         | 147264 ns        | 153397 ns        | 143567 ns        | 150541 ns        |
| RapidJSON            | 1081987 ns       | 1091551 ns       | 1075215 ns       | 1085541 ns       |
| Lite³ Context API    | 2285 ns          | 17820 ns         | 456 ns           | 11869 ns         |
| Lite³ Buffer API     | 2221 ns          | 17659 ns         | 448 ns           | 11699 ns         |

**To be clear: the other formats are parsing JSON.**  
**Lite³ operates on the same dataset, but converted to binary Lite³ format in order to show the potential.**

This benchmark is open source and can be replicated [here](https://github.com/fastserial/simdjson).

### Kostya JSON Benchmark
[A somewhat popular benchmark](https://github.com/kostya/benchmarks) comparing the performance of different programming languages.
In the JSON category, a ~115 MB JSON document is generated consisting of many floating point numbers representing coordinates.
The program will be timed for how long it takes to sum all the numbers.

The aim for this test is similar: quantifying the advantage of a zero-copy format.
This time, reading the entire dataset is unavoidable to produce a correct result.
So instead, the emphasis will be on text-to-binary conversion.
Because Lite³ stores numbers natively in 64 bits, there is no need to parse and convert ASCII-decimals.
This conversion can be tricky for floating point numbers in particular.

![](img/lite3_benchmark_kostya_json_execution_time.png)
![](img/lite3_benchmark_kostya_json_memory_usage.png)

| Language / Library            | Execution Time       | Memory Usage         |
| ----------------------------- | -------------------- | -------------------- |
| C++/g++ (DAW JSON Link)       | 0.094 s              | 113 MB               |
| C++/g++ (RapidJSON)           | 0.1866 s             | 238 MB               |
| C++/g++ (gason)               | 0.1462 s             | 209 MB               |
| C++/g++ (simdjson DOM)        | 0.1515 s             | 285 MB               |
| C++/g++ (simdjson On-Demand)  | 0.0759 s             | 173 MB               |
| C/gcc (lite3)                 | 0.027 s              | 203 MB               |
| C/gcc (lite3_context_api)     | 0.027 s              | 203 MB               |
| Go (Sonic)                    | 0.2246 s             | 121 MB               |
| Rust (Serde Custom)           | 0.113 s              | 111 MB               |
| Zig                           | 0.2493 s             | 147 MB               |

**To be clear: the other formats are parsing JSON.**  
**Lite³ operates on the same dataset, but converted to binary Lite³ format in order to show the potential.**

This benchmark is open source and can be replicated [here](https://github.com/fastserial/kostya-benchmark).

### Cpp Serialization Benchmark
It is to be expected that binary formats will peform well compared to text formats.
The comparison however is not entirely unwarranted.
Pure binary formats present another category, typically requiring schema files and extra tooling.
They are chosen by those who value performance over other considerations.
In doing so, trade-offs are made in usability and flexibility.

Lite³ also being a binary format, rather opts for a schemaless design.
This produces a more balanced set of trade-offs with the notable feature of JSON-compatibility.

Performance of course will remain a strong selling point.
This next benchmark originates from the [Cista++ serialization library](https://github.com/felixguendling/cpp-serialization-benchmark) to compare several binary formats, including zero-copy formats.
The measurements cover the time required to serialize, deserialize and traverse a graph consisting of nodes and edges.
The Cista++ authors created three variants for their format, notably the 'offset' and 'offset slim' variants
where the edges use indices to reference nodes instead of pointers.

![](img/lite3_benchmark_cpp_serialization.png)

| Name                  | Serialize + Deserialize | Deserialize | Serialize   | Traverse    | Deserialize and traverse | Message size    |
| --------------------- |------------------------ | ----------- | ----------- | ----------- | ------------------------ | --------------- |
| Cap’n Proto           | **66.55 ms**            | 0 ms        | 66.55 ms    | 210.1 ms    | 211 ms                   | 50.5093 MB      |
| cereal                | **229.16 ms**           | 98.76 ms    | 130.4 ms    | 79.17 ms    | 180.7 ms                 | 37.829 MB       |
| Cista++ (offset)      | **913.2 ms**            | 274.1 ms    | 639.1 ms    | 79.59 ms    | 80.02 ms                 | 176.378 MB      |
| Cista++ (offset slim) | **3.96 ms**             | 0.17 ms     | 3.79 ms     | 79.99 ms    | 80.46 ms                 | 25.317 MB       |
| Cista++ (raw)         | **947.4 ms**            | 289.2 ms    | 658.2 ms    | 81.53 ms    | 113.3 ms                 | 176.378 MB      |
| Flatbuffers           | **1887.49 ms**          | 41.69 ms    | 1845.8 ms   | 90.53 ms    | 90.35 ms                 | 62.998 MB       |
| Lite³ Buffer API      | **7.79 ms**             | 4.77 ms     | 3.02 ms     | 79.39 ms    | 84.92 ms                 | 38.069 MB       |
| Lite³ Context API     | **7.8 ms**              | 4.76 ms     | 3.04  ms    | 79.59 ms    | 84.13 ms                 | 38.069 MB       |
| zpp::bits             | **4.66 ms**             | 1.9 ms      | 2.76 ms     | 78.66 ms    | 81.21 ms                 | 37.8066 MB      |


This benchmark is open source and can be replicated [here](https://github.com/fastserial/cpp-serialization-benchmark).


## Code Example
Lite³ is a binary format, but the examples print message data as JSON to `stdout` for better readability.

Here is an example with error handling omitted for brevity, taken from `examples/buffer_api/01-building-messages.c`:
```C
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "lite3.h"


static unsigned char buf[1024], rx[1024];

int main() {
        size_t buflen = 0;
        size_t bufsz = sizeof(buf);

        // Build message
        lite3_init_obj(buf, &buflen, bufsz);
        lite3_set_str(buf, &buflen, 0, bufsz, "event", "lap_complete");
        lite3_set_i64(buf, &buflen, 0, bufsz, "lap", 55);
        lite3_set_f64(buf, &buflen, 0, bufsz, "time_sec", 88.427);
        printf("buflen: %zu\n", buflen);
        lite3_json_print(buf, buflen, 0); // Print Lite³ as JSON

        printf("\nUpdating lap count\n");
        lite3_set_i64(buf, &buflen, 0, bufsz, "lap", 56);
        printf("Data to send:\n");
        printf("buflen: %zu\n", buflen);
        lite3_json_print(buf, buflen, 0);
        
        // Transmit
        size_t rx_buflen = buflen;
        size_t rx_bufsz = sizeof(rx);
        memcpy(rx, buf, buflen);
        
        // Mutate (zero-copy, no parsing)
        printf("\nVerifying fastest lap\n");
        lite3_set_str(rx, &rx_buflen, 0, rx_bufsz, "verified", "race_control");
        lite3_set_bool(rx, &rx_buflen, 0, rx_bufsz, "fastest_lap", true);
        printf("Modified data:\n");
        printf("rx_buflen: %zu\n", rx_buflen);
        lite3_json_print(rx, rx_buflen, 0);

        // Ready to send:
        // send(sock, rx, rx_buflen, 0);

        return 0;
}
```
Output:
```
buflen: 154
{
    "lap": 55,
    "event": "lap_complete",
    "time_sec": 88.427
}

Updating lap count
Data to send:
buflen: 154
{
    "lap": 56,
    "event": "lap_complete",
    "time_sec": 88.427
}

Verifying fastest lap
Modified data:
rx_buflen: 197
{
    "lap": 56,
    "event": "lap_complete",
    "time_sec": 88.427,
    "fastest_lap": true,
    "verified": "race_control"
}
```
Lite³ provides an alternative API called the 'Context API' where memory management is abstracted away from the user.

This example is taken from `examples/context_api/04-nesting.c`. Again, with error handling omitted for brevity:
```C
#include <stdio.h>
#include <string.h>

#include "lite3_context_api.h"


int main() {
        lite3_ctx *ctx = lite3_ctx_create();
        
        // Build message
        lite3_ctx_init_obj(ctx)
        lite3_ctx_set_str(ctx, 0, "event", "http_request")
        lite3_ctx_set_str(ctx, 0, "method", "POST")
        lite3_ctx_set_i64(ctx, 0, "duration_ms", 47)

        // Set headers
        size_t headers_ofs;
        lite3_ctx_set_obj(ctx, 0, "headers", &headers_ofs)
        lite3_ctx_set_str(ctx, headers_ofs, "content-type", "application/json")
        lite3_ctx_set_str(ctx, headers_ofs, "x-request-id", "req_9f8e2a")
        lite3_ctx_set_str(ctx, headers_ofs, "user-agent", "curl/8.1.2")

        lite3_ctx_json_print(ctx, 0) // Print Lite³ as JSON

        // Get user-agent
        lite3_str user_agent;
        size_t ofs;
        lite3_ctx_get_obj(ctx, 0, "headers", &ofs)
        lite3_ctx_get_str(ctx, ofs, "user-agent", &user_agent)
        printf("User agent: %s\n", LITE3_STR(ctx->buf, user_agent));

        lite3_ctx_destroy(ctx);
        return 0;
}
```
Output:
```
{
    "method": "POST",
    "event": "http_request",
    "duration_ms": 47,
    "headers": {
        "user-agent": "curl/8.1.2",
        "x-request-id": "req_9f8e2a",
        "content-type": "application/json"
    }
}
User agent: curl/8.1.2
```
For a complete How-to Guide with examples, see [the documentation](https://lite3.io/how_to_guides.html).


## Getting Started
### Make Commands
| Command           | Description                                               |
|-------------------|-----------------------------------------------------------|
| `make all`        | Build the static library with -O2 optimizations (default) |
| `make tests`      | Build and run all tests                                   |
| `make examples`   | Build all examples                                        |
| `make install`    | Install library in `/usr/local` (for pkg-config)          |
| `make uninstall`  | Uninstall library                                         |
| `make clean`      | Remove all build artifacts                                |
| `make help`       | Show this help message                                    |

### Installation
A gcc or clang compiler is required due to the use of various builtins.

First clone the repository:
```
git clone https://github.com/fastserial/lite3.git
cd lite3/
```
Then choose between installation via `pkg-config` or manual linking.

#### Installation via pkg-config (easiest)
Inside the project root, run:
```
sudo make install -j
sudo ldconfig
```
This will build the static library, then install it to `/usr/local` and refresh the `pkg-config` cache. If installation was successful, you should be able to check the library version like so:
```
pkg-config --modversion lite3
```
You can now compile using these flags:
```
$(pkg-config --libs --cflags --static lite3)
```
For example, to compile a single file `main.c`:
```
gcc -o main main.c $(pkg-config --libs --cflags --static lite3)
```
#### Installation via manual linking
First build the library inside project root:
```
make -j
```
Then in your main program:
1. Link against `build/liblite3.a`
2. And include: `include/lite3.h` + `include/lite3_context_api.h`

For example, to compile a single file `main.c`:
```
gcc -o main main.c -I/path/to/lite3/include /path/to/lite3/build/liblite3.a
```
### Using the library
#### Choose your API
The Buffer API provides the most control, utilizing caller-supplied buffers to support environments with custom allocation patterns, avoiding the use of `malloc()`.

The Context API is a wrapper aound the Buffer API where memory allocations are hidden from the user, presenting a more accessible interface. If you are using Lite³ for the first time, it is recommmended to start with the Context API.
```C
#include "lite3.h"              // Buffer API
#include "lite3_context_api.h"  // Context API
```
There is no need to include both headers, only the API you intend to use.

#### Library error messages
By default, library error messages are disabled. However it is recommended to enable them to receive feedback during development. To do this, either:
1. uncomment the line `// #define LITE3_ERROR_MESSAGES` inside the header file: `include/lite3.h`
2. build the library using compilation flag `-DLITE3_ERROR_MESSAGES`

If you installed using `pkg-config`, you may need to reinstall the library to apply the changes. To do this, run:
```
sudo make uninstall
sudo make clean
sudo make install
sudo ldconfig
```
#### Building Examples
Examples can be found in separate directories for each API:
- `examples/buffer_api/*`
- `examples/context_api/*`

To build the examples, inside the project root run:
```
make examples -j
```
To run an example:
```
./build/examples/context_api/01-building-messages
```
For learning how to use Lite³, it is recommended to follow the [How-to Guide series](https://lite3.io/how_to_guides.html).


## Lite³ Explained
### JSON comparison
*tech's so clever and so refined, yet still parses text like it's '99 🎶🎵*
![](img/lite3_chad_meme.jpg)

To understand Lite³, let us first look at JSON.

JSON is a text format. This makes it very convient for humans. We can read it in any text editor, understand and modify it if necessary.

Unfortunately, a computer cannot directly work with numbers in text form. It can display them, sure. But to actually do useful calculations like addition, multiplication etc. it must convert them to binary, because processors can only operate on numbers in native word sizes (8-bit, 16-bit, 32-bit, 64-bit). This conversion from text to binary must be done through parsing. This happens inside your browser when you visit a website, and all around the world, millions or billions of times per second by computers communicating with eachother through all kinds of protocols, APIs etc.

There are broadly 3 strategies or 'ways' to parse JSON:
1. **DOM-based approach**: The Document Object Model (DOM) is a memory representation of a document as a logical tree, of which the contents can be programmatically accessed and modified. When receiving a JSON-message, it is *parsed* into a dictionary object or DOM-tree. This structure acts as the 'intermediate form' to support mutations and lookups. This approach is dominant, because for programmers it is by far the easiest.  Yes, first all the text must be converted into the tree-model, but this can be done automatically by the computer, and once it is done, all of the document's data is directly available and *mutable*. This approach is used by virtually every JSON library and even by your browser to represent HTML web pages. To send data, the tree must be *serialized* again into a JSON string.
2. **Streaming / event-based approach (SAX)**: This approach consists of a pointer iterating over the document. The text is processed from the beginning, and each newly encountered component is an event. For example, the beginning of a string, the beginning of an array, the end of an array etc. The programmer may provide functions corresponding to each event, such as executing `foo(str)` everytime a string is encountered. This approach allows for capturing of only the information that the program needs, ignoring everything else. While more efficient, it is also very inconvenient, because you might discover your program 'missed' an event it now wants to read and has to go back and restart the iterator. The programmer must constantly keep track of their logical position within the document, which could lead to complexity and bugs. Therefore, this approach is not very popular.
3. [**On-Demand approach**](https://arxiv.org/abs/2312.17149): A relatively new technique employed by the [simdjson C++ library](https://github.com/simdjson/simdjson). It resembles a DOM-based approach, except that the parser is 'lazy' and will not parse anything until the very last moment when you actually try to read a value. Unfortunately, some restrictions leak into the abstractions of the user API. For example, [values can only be consumed once](https://github.com/simdjson/simdjson/blob/master/doc/basics.md#using-the-parsed-json) and must be stored separately if you plan to use them multiple times. Therefore it more closely resembles a streaming interface and users might still prefer the DOM-approach for practical reasons. Additionally, the performance of JSON On-Demand is sensitive to the ordering of keys within objects. Many JSON serializers do not preserve or guarantee key ordering in any way. Therefore, it could introduce unexpected variance in read latency. Simdjson also has another major restriction: parsing is read-only. The library does not allow you to modify the JSON document and write it back out.

Despite the DOM-based approach being easiest for the programmer, for the computer it represents a non-trivial amount of work. The text must be parsed to find commas, brackets, decimals etc. A block of memory must be allocated and then the tree must be built. The same data is now duplicated and stored in two different representations: as DOM-tree and string. All these operations add memory overhead, runtime cost and increased latency.

So we spend a lot of time constructing and serializing (separate) DOM-trees. Wouldn't it be great if we could, say, encode the tree directly inside the message?

That is exactly what we do. In Lite³, the DOM-tree index is *embedded directly inside the structure of the message itself*. The underlying datastructure is a B-tree. As a result, lookups, inserts and deletions (`get()`, `set()` & `delete()`) can be performed directly on the serialized form. Encoding to a separate representation becomes unnecessary. Since it is already serialized, it can be sent over the wire at any time. This works because Lite³ structures are always contiguous in memory. Think of the 'dictionary itself' being sent.  
We can perform zero-copy lookups, meaning that we do not need to process the entire message, just the field we are interested in. Similarly, we can insert and delete data only by reading the index and nothing else.

Another side effect of having a full functioning B-tree inside a serialized message is that it is possible to **mutate data directly in serialized form**. Tree rebalancing, key inserts etc. all happen inside the message data. Typically, to mutate serialized data *en route* you would have to:

        Receive JSON string -> Parse -> Mutate -> Stringify -> Send JSON string

With Lite³, this simply becomes:

        Receive Lite³ message -> Mutate -> Send Lite³ message

Many internal datacenter communications consist of patterns where messages are received, a few things modified, then sent on to the other services. In such patterns, the design of Lite³ shines because the reserialization overhead is entirely avoided. It is possible to insert entirely new keys, values, or to overriding existing values while completely avoiding reserialization.

With JSON, if you change just a single field inside a 1 MB document, you typically cannot avoid reserializing the entire 1 MB document. But with Lite³, you call an API function to traverse the message, find the field and change it in-place. Such an operation can be done in tens of nanoseconds on modern hardware given the data is present in cache.

Lite³ also implements a `BYTES` type for native encoding of raw bytes. JSON however does not support this. When converting from Lite³ to JSON, bytes are automatically converted to a base64-encoded string.


### Why B-tree?
Lite³ internally uses a B-tree datastructure. B-trees are well-known performant datastructures, as evidenced by the many popular databases using them (SQLite, MySQL, PostgreSQL, MongoDB & DynamoDB).
The reason for their popularity is that they allow for dynamic insertions/deletions while keeping the datastructure balanced and always guaranteeing `log(n)` lookups.

However, B-trees are rarely found in a memory or serialization context. For memory-only datastructures, common wisdom rather opts for hashmaps or classic binary trees. B-trees are seen as too 'heavyweight'. But we can do the same in memory, though we need a much more compact 'micro B-tree' specifically adapted for fast memory operations. Databases typically use 4kB to store a node in the tree, matching disk pages sizes. Since we are in memory, we work with cache lines, not disk pages. Therefore the node size in Lite³ is set to a (configurable) 96 bytes or 1.5 cache lines by default. [Literature suggests](https://arxiv.org/abs/2503.23397) that modern machines have nearly identical latency for 64-byte and 256-byte memory accesses, though larger nodes will also increase message size. CPU performance in the current age is all about access patterns and cache friendliness. It may surprise some people that as a result, memory B-trees [actually outperform](https://en.algorithmica.org/hpc/data-structures/s-tree/) classic binary trees and red-black trees.

An algorithmic observer will note that B-trees have logarithmic time complexity, versus hashmaps' constant time lookups. But typical serialized messages are often small ([70% of JSON messages are < 10kB](https://www.researchgate.net/figure/Size-in-bytes-for-JSON-and-XML-payloads-and-media-type-distribution-by-host_fig2_303515127)) and only read once. So the overhead of other operations will dominate, except with very frequent updates on large structures (larger than LLC). More importantly, using Lite³ structures completely eliminates an `O(n)` parsing and serialization step.

### Byte Layout
In Lite³, all tree nodes and data entries are serialized inside a single byte buffer.
Tree hierarchies are established using pointers; all parent nodes have pointers to their child nodes:
```
|----------------|-------|----------------|----------------|-------------|----------------|-------------
|      ROOT      |#DATA##|    CHILD #1    |    CHILD #2    |#####DATA####|    CHILD #3    |####DATA####...
|----------------|-------|----------------|----------------|-------------|----------------|-------------
   | | |___________________|                |                              |
   | |______________________________________|                              |
   |_______________________________________________________________________|
```

Nodes consist of key hashes stored alongside pointers to key-value pairs (data entries), as well as child node pointers:
```
 NODE
|--------------------------------------------------------------------------------------------|
|      h(key1) h(key2) h(keyN)      *kv1 *kv2 *keyN      *child1 *child2 *child3 *childN     |
|--------------------------------------------------------------------------------------------|
       |                            |                    |
      hashed keys                  kv-pair pointers     child node pointers
      used for key comparison      pointing to data     used for tree traversal
                                   entries
```

The key-value pairs are stored anywhere between nodes, or at the end of the buffer:
```
 KEY-VALUE PAIR              
|--------|-----------|-------------------------|
|  type  |    key    |          value          |
|--------|-----------|-------------------------|
   |          |
   1 byte     string
   type tag   key
```
- On insertion: The new nodes or key-value pairs are appended at the end of the buffer.
- On deletion: The key-value pairs are deleted in place.
               TODO: freed blocks are added to the GC (garbage collection) index.

Over time, deleted entries cause the contiguous byte buffer to accumulate 'holes'. For a serialization format, this is undesirable as data should be compact to occupy less bandwidth. Lite³ will use a defragmentation system to lazily insert new values into the gaps whenever possible. This way, fragmentation is kept under control without excessive data copying or tree rebuilding (STILL WIP, NOT YET IMPLEMENTED).

>    NOTE: By default, deleted values are overwritten with NULL bytes (0x00). This is a safety feature since not doing so would leave 'deleted' entries intact inside the datastructure until they are overwritten by other values.
>    If the user wishes to maximize performance at the cost of leaking deleted data, `LITE3_ZERO_MEM_DELETED` should be disabled.

Also, Lite³ does not store raw pointers, but rather 32-bit indexes relative to the buffer pointer. The buffer pointer always points to the zero-index, and the root node is always stored at the zero-index.

Despite being a binary format, Lite³ is schemaless and can be converted to/from JSON. For a more complete explanation of the design, see [Design and Limitations](https://lite3.io/design_and_limitations.html).


## Other Serialization Formats
### 'Binary JSON' formats
A number of formats advertise themselves as being 'binary JSON'. Instead of bracks and commas, they typically use a system of tags (TLV) to encode types and values. Being binary also means they store numbers natively, avoiding the parsing of ASCII floating point decimals which is known to be performance-problematic for text formats.

The notable contenders are:
- **BSON**: Or 'binary JSON', known for being the underlying format in the WiredTiger storage engine of MongoDB. It supports various native types such as `Date` objects.
- **Amazon Ion**: A JSON-superset format used by Amazon with a richer type system, similar to BSON.
- **MsgPack**: Very fast and small format. Can achieve good performance depending on implementation.
- **CBOR**: Inspired by MessagePack but with IETF standardization and extended types.
- **Smile**: A format that implements back-referencing (deduplication) to achieve smaller message sizes compared to JSON. Used in ElasticSearch.
- **UBJSON**: Aims to achieve the generality of JSON while simplifying parser implementations.

While these formats avoid the conversion of numbers from text, they do not eliminate parsing entirely. All of these formats still require a separate memory representation such as a DOM-tree to support meaningful mutations, including reserializing overhead.

Fundamentally, this flaw arises out the fact that values are stored contiguously like arrays, meaning they suffer from all the downsides of arrays. To find an element inside, typically an `O(n)` linear search is required. This is particularly problematic for random access on large element counts. Additionally, the contiguous nature means that a change to an internal element will require (partial) rewriting of the document.

In contrast, Lite³ is a zero-copy format storing all internal elements within a B-tree structure, guaranteeing `O(log n)` amortized time complexity for access and modification of any internal value, even inside arrays. The modification of an internal element will also never trigger a rewrite of the document. Only the target element might require reallocation and updating of the corresponding reference. Even throughout modifications, zero-copy access is maintained.

Therefore these formats may be interesting from a perspective of compactness or rich typing. However looking from the standpoint of encode/decode performance, they exist in a lower category.

### Schema-only Binary Formats
By making a number of assumptions about the structure of a serialized message, it is possible to (greatly) accelerate the process of encoding/decoding to/from the serialized form. This is where so-called 'binary formats' come in.

But if binary formats exist and are much faster, why is everyone still using JSON?

The answer lies in the fact that most binary formats require a schema file written in an IDL; basically an instruction manual for how to read a message. A binary message doesn't actually tell you anything. It is literally just a bunch of bytes. Only with the schema does it acquire meaning. Note that 'binary' does not necessarily mean schema-only, though in practice this is often implied.

When sending messages between systems you control, you can create your own schemas. But communicating with other people's servers? Now you need to use their schemas as well. And if you want to communicate with the whole world? Well you better start collecting schemas. Relying on out-of-band information eventually takes its toll. Imagine needing an instruction manual for every person you wanted to talk to. Crazy right?

Because of these restrictions, schema-only formats reside in their own special category, notably distinct from *schemaless* and self-describing formats like JSON, which can be directly read and interpreted without the requirement of extra outside information.

That said, these formats come in 3 primary forms:
1. **Static code generation**:
      This is the approach taken by Protocol Buffers. Schema definition files (`.proto`) are compiled using the external `protoc` Protobuf compiler into encoding/decoding source code (C++, Go, Java) for each type of message. This is also the [approach taken by Rust's "serde" crate](https://docs.rs/serde/latest/serde/#design), although the compilation is performed by the Rust compiler, not an external tool.
2. **Compile-time reflection**:
      With 'reflection' (a.k.a. introspection), the compiler inspects the code's own structure during compilation to generate the serialization logic, avoiding the need for a separate schema file.
      This approach is used by C++ libraries such as [Glaze](https://github.com/stephenberry/glaze) and [cpp-reflect](https://github.com/getml/reflect-cpp). Furthermore, [C++ is getting reflection in C++26](https://lemire.me/blog/2024/08/13/reflection-based-json-in-c-at-gigabytes-per-second/). However, these libraries only support serialization to/from C++ classes. There is no cross-language support.
3. **Runtime reflection**:
      In C#, [the JsonSerializer collects metadata at run time by using reflection](https://learn.microsoft.com/en-us/dotnet/standard/serialization/system-text-json/reflection-vs-source-generation). Whenever it has to serialize or deserialize a type for the first time, it collects and caches this metadata.
      This approach resembles a JIT-like way to accelerate 'learned' message schema. But the downside is that the metadata collection process takes time and uses memory.  
      Other environments such as the V8 JavaScript engine employ a 'fast path' for JSON serialization if the message conforms to [specific rules](https://v8.dev/blog/json-stringify). Otherwise, the traditional 'slow path' must be taken.

Schema-only formats tend to be brittle and require simultaneous end-to-end upgrades to handle change.
Although backwards-compatible evolution is possible, it requires recompilation and synchronization of IDL specifications.
But updating all clients and servers simultaneously can be challenging. Major changes like renaming, removing fields or changing types can lead to silent data loss or incompatibility if not handled correctly. In some cases it is better to define a new API endpoint and deprecate the old one.

Here is a section taken from the [Simple Binary Encoding's "Design Principles" page](https://github.com/aeron-io/simple-binary-encoding/wiki/Design-Principles):
> Backwards Compatibility  
> In a large enterprise, or across enterprises, it is not always possible to upgrade all systems at the same time. For communication to continue working the message formats have to be backwards compatible, i.e. an older system should be able to read a newer version of the same message and vice versa.
> 
> An extension mechanism is designed into SBE which allows for the introduction of new optional fields within a message that the new systems can use while the older systems ignore them until upgrade. If new mandatory fields are required or a fundamental structural change is required then a new message type must be employed because it is no longer a semantic extension of an existing message type.

For self-describing formats like Lite³ and JSON, adding and removing fields is easy. It is only positional formats that have this problem. More precisely:
- **Positional formats**: Require backwards compatibility on the ABI layer + application layer.
- **Self-describing formats**: Require backwards compatibility only on the application layer.

### Zero-Copy Formats
There sometimes exists ambiguity and confusion around the term 'zero-copy'. What does it mean for a data format to be 'zero-copy', or any system in general?

In most contexts, zero-copy refers to **a method of accessing and using data without physically moving or duplicating it from its original location**. The guiding principle being that compute resources should be spent on real work or calculations, not wasting CPU cycles on unnecessarily relocating and moving bytes around. Applications, operating systems and programming languages may all use zero-copy techniques in various forms, under various names:
- **C-like languages**: Taking pointers into buffers instead of passing around entire values
- **Linux kernel**: `splice()` and `sendfile()` system calls
- **Rust**: `&[u8]` slices, `std::io::Cursor` and `bytes::Bytes`
- **C++**: `std::string_view` and `std::span`
- **Python**: `memoryview` object
- **Java**: `java.nio.ByteBuffer` and `slice()`
- **C#**: `Span<T>`, `Memory<T>` and `ReadOnlySpan<T>`
- **Go**: Slices (e.g., `mySlice := myArray[1:4]`)
- **JavaScript / Node.js**: `Buffer.subarray()`, `TypedArray` views on an `ArrayBuffer`

Moving around data is a real cost. In the best case, performance degrades linearly with the size of the data being moved. In reality, memory allocations, cache misses, garbage collection and other overhead mean that these costs can multiply non-linearly.

When we talk about 'zero-copy serialization formats', the format should support reading values from the original location, i.e. directly from the serialized message. If a format requires thats its contents are first transformed into an alternative memory representation (i.e. DOM-tree), then this does not classify as zero-copy.

In some cases, a format may support 'zero-copy' references to string or byte objects. However, the ability to access some member field by reference does not immediately make a format 'zero-copy'. Instead, every member field must be accessible without requiring any parsing or transformation step on a received message.

The 4 most notable existing zero-copy formats are:
1. **Flatbuffers (by Google)**:
        Mainly used in game development or performance heavy applications.
        One caveat is that while decoding Flatbuffers is very fast, encoding is much slower. In fact, modern JSON libraries can beat it at encoding.
        Also, Flatbuffers has been critized for having an awkward API. This is because logically, the data needs to be encoded inside-out. First all the children/leaf sizes must be known before the parent can be encoded.
2. **Flexbuffers (by Google)**:
        The schemaless version of Flatbuffers, made by the same author. It cannot match the instant decoding speed of Flatbuffers. However, being schemaless it requires no schema definitons. Though it never received much popularity or adoption.
3. **Cap'n Proto**:
        [Similar to Flatbuffers](https://capnproto.org/news/2014-06-17-capnproto-flatbuffers-sbe.html), but using a more standard builder API. Despite having good performance, it sees limited use in real applications and [its author is only semi-actively maintaining it](https://news.ycombinator.com/item?id=40391670). Another downside is that optional fields take space over the wire, even when unused.
4. **SBE (Simple Binary Encoding)**:
        Mainly used in low-latency financial applications (realtime trades & price data). The format was designed specifically for this purpose and resembles little more than a struct sent over a network. It requires schema definitions written in XML.

Note that all of these formats except Flexbuffers require rigid, pre-defined schema compiled into your application.
Also, none of these formats support arbitrary mutation of serialized data. If a single field must be changed, then the entire message must be re-serialized. Only the latter 2 formats support trivial in-place mutation of fixed-sized values.


## Worldwide Impact
As of writing, [JSON](https://www.rfc-editor.org/rfc/rfc7493.html) remains the global standard today for data serialization. Reasons include: ease of use, human readability and interopability.

Though it comes with one primary drawback: performance. When deploying services at scale using JSON, parsing/serialization can become a serious bottleneck.

The need for performance is ever-present in today's world of large-scale digital infrastructure. For parties involved, cloud and electricity costs are significant factors which cannot be ignored.
[Based on a report by the IEA](https://www.iea.org/data-and-statistics/charts/data-centre-electricity-consumption-by-region-base-case-2020-2030), data centres in 2024 used 415 terawatt hours (TWh) or about 1.5% of global electricity consumption. This is expected to double and reach 945 TWh by 2030.

Building systems that scale to millions of users requires being mindful of cloud costs. [According to a paper from 2021](https://dl.acm.org/doi/10.1145/3466752.3480051), protobuf operations constitute 9.6% of fleet-wide CPU cycles in Google’s infrastructure.
Microservices at Meta (Facebook) [also spend](https://dl.acm.org/doi/10.1145/3373376.3378450) between 4-13% of CPU cycles on (de)serialization alone.
Similar case studies of [Atlassian](https://www.atlassian.com/blog/atlassian-engineering/using-protobuf-to-make-jira-cloud-faster) and [LinkedIn](https://www.linkedin.com/pulse/performance-optimization-linkedins-shift-from-json-protocol-saxena-mgr4c) show the need to step away from JSON for performance reasons.

JSON is truly widespread and ubiquitous. If we estimate that inefficient communication formats account for 1-2% of datacenter infrastructure, this amounts to several TWh annualy; comparable to the energy consumption of a small country like Latvia (7.17 TWh in 2023) or Albania (8.09 TWh in 2023).
True figures are hard to obtain, but for a comprehensive picture, all devices outside datacenters must also be considered.
Not just big tech, but also hardware devices, IoT and a myriad of other applications across different sectors have spawned a variety of 'application specific' binary formats to answer the performance question.

But many binary formats are domain specific. Or they require rigid schema definitions, typically written using some IDL and required by both sender and receiver. Both must be in sync at all times to avoid communication errors.
Then if the schema should be changed (so-called 'schema evolution'), it is often a complex and fragile task to preserve backwards compatibility.
This, combined with lacking integration in web browsers means many developers avoid binary formats despite performance benefits.

*Purely schemaless* formats are simply easier to work with. This fact is evidenced by the popularity of JSON.
For systems talking to eachother, fragmented communications and lack of standards become problematic, especially when conversion steps are required between different formats. In many cases, systems still fall back to JSON for interopability.

Despite being schemaless, Lite³ directly competes with the performance of binary formats.


## Security
Lite³ is designed to handle untrusted messages. Being a pointer chasing format, special attention is paid to security. Some measures include:
- Pointer dereferences preceded by overflow-protected bounds checks.
- Runtime type safety.
- Max recursion limits.
- Generational pointer macro to prevent dangling pointers into Lite³ buffers.

If you suspect to have found a security vulnerability, please [contact the developer](mailto:elias@fastserial.com).


## Q&A
**Q: Should I use this instead of JSON in my favorite programming language?**  
A: If you care about performance and can directly interface with C code, then go ahead. If not, wait for better language bindings.

**Q: Should I use this instead of Protocol Buffers (or any other binary format)?**  
A: In terms of encode/decode performance, Lite³ outperforms Protobuf due to the zero-copy advantage. But Lite³ must encode field names to be self-describing, so messages take up more space over the wire. So choose Lite³ if you are CPU-constrained. Are you bandwidth constrained? Then choose Protocol Buffers and be prepared to accept extra tooling, IDL and ABI-breaking evolution to minimize message size.

**Q: Can I use this in production?**  
A: The format is developed for use in the field, though keep in mind this is a new project and the API is unstable. Also: understand [the limitations](https://lite3.io/design_and_limitations.html). Experiment first and decide if it suits your needs.

**Q: Can I use this in embedded / ARM?**  
A: Yes, but your platform should support the `int64_t` type, 8-byte doubles and a suitable C11 gcc/clang compiler,
though downgrading to C99 is possible by removing all static assertions.
The format has not yet been tested on ARM.


## Roadmap
- [ ] Optimize build and add support for `-flto`
- [ ] Built-in defragmentation with GC-index
- [x] Full JSON interoperability with arrays & nested objects
- [x] Opt-out compilation flag for `yyjson`
- [ ] Handling key collisions
- [ ] Size benchmark for compression ratios using different codecs
- [ ] Add language bindings
- [ ] Write formal spec


## Mailing List
If you would like to be part of developer discussions with the project author, consider joining the mailing list:

`devlist@fastserial.com`        

To join, [send a mail](mailto:devlist-subscribe@fastserial.com) to `devlist-subscribe@fastserial.com` with non-empty subject.
You will receive an email with instructions to confirm your subscription.

Reply is set to the entire list, though with moderation enabled.

To quit the mailing list, simply mail `devlist-unsubscribe@fastserial.com`


## Credit
This project was inspired by a paper published in 2024 as *Lite²*:
> Tianyi Chen †, Xiaotong Guan †, Shi Shuai †, Cuiting Huang † and Michal Aibin †  
(2024).  
Lite²: A Schemaless Zero-Copy Serialization Format  
https://doi.org/10.3390/computers13040089

The paper authors in turn got their idea from SQL databases. They noticed how it is possible to insert arbitrary keys, therefore being schemaless. Also, performing a key lookup can be done without loading the entire DB in memory, thus being zero-copy.  
They theorized that it would be possible to remove all the overhead associated with a full-fledged database system, such that it would be lightweight enough to be used as a serialization format. They chose the name *Lite²* since their format is lighter than SQLite.  
Despite showing benchmarks, the paper authors did not include code artifacts.

The Lite³ project is an independent interpretation and implementation, with no affiliations or connections to the authors of the original Lite² paper.

## The Lite³ name
The name **Lite³** was chosen since it is lighter than Lite².
> TIP: To type `³` on your keyboard on Linux hold `Ctrl`+`Shift`+`U` then type `00B3`. On Windows, use `Alt`+(numpad)`0179`.


## License
Lite³ is released under the MIT License. Refer to the LICENSE file for details.

For JSON conversion, Lite³ also includes `yyjson`, the fastest JSON library in C.
`yyjson` is written by YaoYuan and also released under the MIT License.

@tableofcontents