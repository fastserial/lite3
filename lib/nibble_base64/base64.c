#include "base64.h"

// MAPS
// b64 maps 0=>A, 1=>B..63=>/ etc
//                      ----------1---------2---------3---------4---------5---------6---
//                      0123456789012345678901234567890123456789012345678901234567890123
static const char* b64="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/" ;

// unb64 maps A=>0, B=>1.. and all others not present in base64 alphabet to 0.
// You can clearly see here why base64 encoding is a really bloated representation
// of the original data: look how many entries are unused. Each ascii character
// can index any value between 0-255 in an array, but we're only using 64 of
// the available slots for meaningful values, leaving 192/256 values unused.
static const unsigned char unb64[]={
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //10 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //20 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //30 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //40 
  0,   0,   0,  62,   0,   0,   0,  63,  52,  53, //50 
 54,  55,  56,  57,  58,  59,  60,  61,   0,   0, //60 
  0,   0,   0,   0,   0,   0,   1,   2,   3,   4, //70 
  5,   6,   7,   8,   9,  10,  11,  12,  13,  14, //80 
 15,  16,  17,  18,  19,  20,  21,  22,  23,  24, //90 
 25,   0,   0,   0,   0,   0,   0,  26,  27,  28, //100 
 29,  30,  31,  32,  33,  34,  35,  36,  37,  38, //110 
 39,  40,  41,  42,  43,  44,  45,  46,  47,  48, //120 
 49,  50,  51,   0,   0,   0,   0,   0,   0,   0, //130 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //140 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //150 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //160 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //170 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //180 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //190 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //200 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //210 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //220 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //230 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //240 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //250 
  0,   0,   0,   0,   0,   0, 
}; // This array has 256 elements

// Converts any binary data to base64 characters.
// Length of the resultant string is stored in flen
// (you must pass pointer flen). flen is strlen, it doesn't
// include the null terminator.
// A common trip-up is not passing your null terminator into
// this function, if you want your C string to be fully encoded,
// you have to pass strlen(str)+1 to as binaryData as I have in the
// examples.
char* nibble_base64( const void* binaryData, int len, int *flen )
{
  //printf("Base64 encoding %d bytes of binary data\n", len);
  
  // I look at your data like the stream of unsigned bytes that it is
  const unsigned char* bin = (const unsigned char*)binaryData ;
  
  int lenMod3 = len % 3;
  
  // PAD. Base64 is all about breaking the input into SEXTETS, or 6-bit inputs.
  // If you have 1 byte of input, that's 8 bits, not 6. So it won't fit.
  // We need the input to be a multiple of 6. So 8 bits would be padded
  // by 2 bytes to make the total input size 24 bits, which is divisible by 6.
  // A 2 byte input is 16 bits, which is not divisible by 6. So we pad it
  // by 1 byte to make it 24 bits, which is now divisible by 6.
  
  // We use modulus 3 bytes above because that's 24 bits, and 24 bits is
  // the lowest number that is both divisible by 6 and 8. We need the final
  // output data is to both be divisible by 6 and 8.
  int pad = ((lenMod3&1)<<1) + ((lenMod3&2)>>1) ; // 2 gives 1 and 1 gives 2, but 0 gives 0.
  
  *flen = 4*(len + pad)/3 ; // (len+pad) IS divisible by 3
  // So, final length IS a multiple of 4 for a valid base64 string.
  //printf("%d %% 3 = %d, %d bytes pad, +1 byte NULL, flen=%d\n", len, lenMod3, pad, *flen);
  
  // Allocate enough space for the base64 string result.
  char* base64String = (char*)malloc( *flen + 1 ) ; // and one for the null,
  // which is NOT counted in flen.
  if( !base64String )
  {
    puts( "ERROR: base64 could not allocate enough memory." ) ;
    puts( "I must stop because I could not get enough" ) ;
    return 0;
  }
  
  // EXTRACTING SEXTETS FROM THE OCTETS.
  //     byte0       byte1       byte2
  // +-----------+-----------+-----------+
  // | 0000 0011   0111 1011   1010 1101 |
  // +-AAAA AABB   BBBB CCCC   CCDD DDDD
  // 
  // In 3 bytes (really, 3 "octets") there are __4__ sextets.
  // You can see that from the diagram above. byte0 (0000 0011) contains
  // the first sextet (AAAA AA) and 2 bits of the 2nd sextet (BB).
  // byte1 contains the next 4 bits of the 2nd sextet (BBBB) and 4 bits
  // of the 3rd sextet (CCCC). byte2 has 2 bits of the 3rd sextet and
  // all 6 bits of the 4th sextet.
  //
  // You can see why we process in groups of 3 bytes: because 3*8 = 24
  // and 24 is the lowest common multiple between 6 and 8. To divide
  // a group of bytes EVENLY into groups of 6, the number of bytes has to
  // be a multiple of 3.
  
  // Talking in bits, the input already HAS to be a multiple of 8 (because you just
  // can't have anything smaller than a byte saved to memory or disk on modern
  // computers). To successfully convert the bitstream into groups of 6 bits, we'll force
  // the input bitstream to being a MULTIPLE OF 24, so that it will evenly
  // divide by 6.
  
  // For that reason, we have the concept of PADDING: if the original octet
  // stream is NOT a multiple of 3, then we pad it with 1 or 2 extra bytes
  // so that it is a multiple of 3.
  
  // So without further ado let's extract the 4 sextets from the 3 octets!
  // Convert sextets in stream into the base64 alphabet using b64 array
  // the value in 6 bits can never be larger than 63, but the b64 array
  // protects us from OOB accesses anyway by providing 
  
  // We want to shift the first 6 bits in the above diagram down to sitting
  // flushed to the right. So we want bin[0] (containing AAAA AABB) to just
  // become 00AA AAAA. We do that with a shift right of 2 bits.
  
  // We take the number that comes out of that and immediately convert it to
  // the base64 character for that number by doing a direct lookup into the
  // b64 encoding array.
  
  // We devise 4 formulae below, SEXTET1, SEXTET2, SEXTET3 and SEXTET4. They
  // are used to extract the 4 sextets from the 3 octets that we have.
  #define SEXTET_A(byte0) (byte0 >> 2)
  // Note that no mask needed since BYTE0 is unsigned, so 0's always come in from left
  // (even though there is implicit int promotion on R&L sides prior to actual bitshift).
  
  // the second sextet BBBBBB is part of the first byte and partly in the 2nd byte.
  //   BYTE0       BYTE1
  // AAAA AABB   BBBB CCCC
  // The first part takes the lower 2 bits of the first byte and pushes them
  // LEFT 4: (AAAA AABB becomes 00BB 0000), then bitwise ORs to it the top 4 bits of
  // BYTE1, shifted RIGHT 4 (BBBB CCCC becomes 0000 BBBB).
  #define SEXTET_B(byte0, byte1) (((0x3&byte0) << 4) | (byte1 >> 4))
  
  // 3rd sextet CCCCCC is lower nibble of 2nd byte and upper half nibble of 3rd byte.
  //   BYTE1       BYTE2
  // BBBB CCCC   CCDD DDDD
  // From BYTE1, we need to get rid of the BBBB in the front, so we mask
  // those off with 0xf (0000 1111). Then we shift BYTE1 LEFT 2
  // (BBBB CCCC becomes 00CC CC00).
  // We need to fill in the bottom 2 bits of 00CC CC00 with the top 2 bits
  // in BYTE2. So we just shift BYTE2 right by 6 bits (CCDD DDDD becomes 0000 00CC).
  #define SEXTET_C(byte1, byte2) (((0xf&byte1) << 2) | (byte2 >> 6))
  
  // 4th sextet
  // already low order, just mask off 2 hiorder bits
  //   BYTE2
  // CCDD DDDD
  // We just want to mask off the top 2 bits, use mask 0011 1111 or just 0x3f
  #define SEXTET_D(byte2) (0x3f&byte2)
  
  int i = 0, byteNo; // result counter, and which byte we're on of the original source data.
  // I still need these variables after the loop
  for( byteNo = 0 ; byteNo <= len-3 ; // This loop is NOT entered for if there
    // are trailing bytes that are not a multiple of 3 bytes,
    // since we skip in 3's.
    // If there WAS padding, skip the last 3 octets and process below.
    // 0=>no, 1=>no, 2=>no, 3=>ONCE,4=>ONCE,5=>ONCE, 6=>2x..
    byteNo+=3 ) // jump in 3's
  {
    // Use unsigned char so shifts left will always bring in 0's
    unsigned char BYTE0 = bin[byteNo];
    unsigned char BYTE1 = bin[byteNo+1];
    unsigned char BYTE2 = bin[byteNo+2];
    //printf( "BYTE0=%d BYTE1=%d BYTE2=%d\n", BYTE0, BYTE1, BYTE2 ) ;
    
    // To form the base64String, we make lookups with the base64 numeric
    // values into the base64 "alphabet" that is present in the b64 array.
    base64String[i++] = b64[ SEXTET_A(BYTE0) ];
    base64String[i++] = b64[ SEXTET_B(BYTE0, BYTE1) ];
    base64String[i++] = b64[ SEXTET_C(BYTE1, BYTE2) ];
    base64String[i++] = b64[ SEXTET_D(BYTE2) ];
  }
  
  // The last 3 octets must be converted carefully as if len%3==1 or len%3==2 we must
  // "pretend" there are additional bits at the end.
  if( pad==1 )
  {
    unsigned char BYTE0 = bin[byteNo];
    unsigned char BYTE1 = bin[byteNo+1];
    //printf( "BYTE0=%d BYTE1=%d\n", BYTE0, BYTE1 ) ;
    // When len%3==2 (2,5,8,11) (missing 1 byte).
    //   - 3 sextets (C is 0 padded)
    //    bin[0]       bin[1]      bin[2]
    // +-----------+-----------+-----------+
    // | 0000 0011   1111 1111   ~~~~ ~~~~ |
    // +-AAAA AABB   BBBB CCCC   XXXX XXXX  
    // Here all the ~ are actually going to be considered __0__'s.
    base64String[i++] = b64[ SEXTET_A(BYTE0) ] ;
    base64String[i++] = b64[ SEXTET_B(BYTE0, BYTE1) ] ;
    
    // We can't use the SEXTET3 formula because we only have 2 bytes to work
    // with. The 3rd byte (BYTE2) is actually 0 here. You could call
    // SEXTET3(BYTE1, 0), but to save some ops we just write what will actually
    // be needed here only.
    base64String[i++] = b64[ (0xf&BYTE1) << 2 ] ;
     
    // Last one is = to indicate there has been a padding of 1 byte.
    base64String[i++] = '=';
  }
  else if( pad==2 ) //len%3==1 (1,4,7,10)
  {
    unsigned char BYTE0 = bin[byteNo];
    //printf( "BYTE0=%d\n", BYTE0 ) ;
    // We are missing 2 bytes. So
    //   - we will only extract 2 sextets when len%3==1
    //   - The 2nd sextet's 2 HI ORDER BITS, NOT LO-ORDER.
    //   - are being specified by the lowest 2 bits of the 1st octet. these should be 0.
    //    bin[0]       bin[1]      bin[2]
    // +-----------+-----------+-----------+
    // | 0000 0011   ~~~~ ~~~~   ~~~~ ~~~~ |
    // +-AAAA AABB   XXXX XXXX   XXXX XXXX
    base64String[i++] = b64[ SEXTET_A(BYTE0) ] ;
    base64String[i++] = b64[ (0x3&BYTE0) << 4 ] ; // "padded" by 0's, these 2 bits are still HI ORDER BITS.
    // Last 2 are ==, to indicate there's been a 2 byte-pad
    base64String[i++] = '=';
    base64String[i++] = '=';
  }
  
  base64String[i] = 0; // NULL TERMINATOR! ;)
  return base64String;
}

unsigned char* nibble_unbase64( const char* ascii, int len, int *flen )
{
#ifdef NIBBLE_SAFEBASE64
  if( !nibble_base64integrity( ascii, len ) )  return 0 ; // NULL PTR if bad integrity.
#endif
  
  const unsigned char *safeAsciiPtr = (const unsigned char*)ascii ; // internally I use
  // an unsigned char pointer, so that __the maximum value read out is 255,
  // and the value is never negative__.  This is a type of
  // "if statement" enforced by the type of the pointer.
  // This eliminates a possible bounds check on your array lookups into unb64[]
  // (*(unsigned char*) having values between 0 and 255 means it will always be
  // inside the bounds of the 256 element array).
    
  int pad = 0;
  if( len > 1 ) {
    // Count == on the end to determine how much it was padded.
    if( safeAsciiPtr[ len-1 ]=='=' )  ++pad;
    if( safeAsciiPtr[ len-2 ]=='=' )  ++pad;
  }
  
  // You take the ascii string len and divide it by 4
  // to get the number of 3 octet groups. You then *3 to
  // get #octets total.
  // If len<4, we makes sure you get a flen of 0, because that's not even
  // a valid base64 string at all.
  *flen = 3*(len/4) - pad;
  if( *flen < 0 )  *flen = 0;
  unsigned char *bin = (unsigned char*)malloc( *flen );
  if( !bin )
  {
    puts( "ERROR: unbase64 could not allocate enough memory." ) ;
    puts( "I must stop because I could not get enough" ) ;
    return 0;
  }
  
  int cb = 0; // counter for bin
  int charNo; // counter for what base64 char we're currently decoding
  
  // NEVER do the last group of 4 characters if either of the
  // last 2 chars were pad.
  for( charNo=0; charNo <= len - 4 - pad ; charNo+=4 )
  {
    // Get the numbers each character represents
    // Since ascii is ONE BYTE, the worst that can happen is
    // you get a bunch of 0's back (if the base64 string contained
    // characters not in the base64 alphabet).
    // The only way unbase64 will TELL you about this though
    // is if you #define NIBBLE_SAFEBASE64 (particularly because
    // there is a 3-4x performance hit, just for the integrity check.)
    int A=unb64[safeAsciiPtr[charNo]];
    //printf( "[%4d] %c => %d\n", charNo, ascii[charNo], A ) ;
    int B=unb64[safeAsciiPtr[charNo+1]];
    //printf( "[%4d] %c => %d\n", charNo+1, ascii[charNo+1], B ) ;
    int C=unb64[safeAsciiPtr[charNo+2]];
    //printf( "[%4d] %c => %d\n", charNo+2, ascii[charNo+2], C ) ;
    int D=unb64[safeAsciiPtr[charNo+3]];
    //printf( "[%4d] %c => %d\n", charNo+3, ascii[charNo+3], D ) ;
    
    // Just unmap each sextet to THE NUMBER it represents.
    // You then have to pack it in bin,
    // we go in groups of 4 sextets, 
    // and pull out 3 octets per quad of sextets.
    //    bin[0]       bin[1]      bin[2]
    // +-----------+-----------+-----------+
    // | 0000 0011   0111 1011   1010 1101 |
    // +-AAAA AABB   BBBB CCCC   CCDD DDDD
    // or them
    bin[cb++] = (A<<2) | (B>>4) ; // OR in last 2 bits of B
    
    // The 2nd byte is the bottom 4 bits of B for the upper nibble,
    // and the top 4 bits of C for the lower nibble.
    bin[cb++] = (B<<4) | (C>>2) ;
    bin[cb++] = (C<<6) | (D) ; // shove C up to top 2 bits, or with D
  }
  
  // If the length of the string were not a multiple of 4, then the string
  // was damaged and some data was lost.
  if( nibble_isMultipleOf(len, 4) )
  {
    if( pad==1 )
    {
      // 1 padding character.
      //    bin[0]       bin[1]      bin[2]
      // +-----------+-----------+-----------+
      // | 0000 0011   1111 1111   ~~~~ ~~~~ |
      // +-AAAA AABB   BBBB CCCC   XXXX XXXX  
      // We can pull 2 bytes out, not 3.
      // We have __3__ characters A,B and C, not 4.
      int A=unb64[safeAsciiPtr[charNo]];
      int B=unb64[safeAsciiPtr[charNo+1]];
      int C=unb64[safeAsciiPtr[charNo+2]];
      
      bin[cb++] = (A<<2) | (B>>4) ;
      bin[cb++] = (B<<4) | (C>>2) ;
    }
    else if( pad==2 )
    {
      //    bin[0]       bin[1]      bin[2]
      // +-----------+-----------+-----------+
      // | 0000 0011   ~~~~ ~~~~   ~~~~ ~~~~ |
      // +-AAAA AABB   XXXX XXXX   XXXX XXXX
      int A=unb64[safeAsciiPtr[charNo]];
      int B=unb64[safeAsciiPtr[charNo+1]];
      
      bin[cb++] = (A<<2) | (B>>4) ;
    }
  }
  
  return bin ;
}

// There are some invalid unbase64 strings, even when they are comprised
// of completely valid characters. An example is "==". That's a 0-length
// piece of data that says it is padded by 2 bytes at the end. Well, you
// only need to pad by 2 bytes if the number of bits in the original data
// was not evenly divisible by 6. 0%6==0, so something's clearly wrong here.
int nibble_base64integrity( const char *ascii, int len )
{
  // The base64 string is somewhat inflated, since each ASCII character
  // represents only a 6-bit value (a sextet). That leaves 2 bits wasted per 8 bits used.
  // More importantly, for the sextet stream you're getting here (inside
  // an octet stream) to be VALID, THE LENGTH HAS TO BE A MULTIPLE OF 4.
  // You can see in the base64 function above, the algorithm always writes
  // into the final base64 string in groups of __4__.
  
  // So from there, you can see a valid base64 string has just gotta have a
  // length that is a multiple of 4.
  
  // If it does not, then it simply isn't valid base64 and the string should
  // be rejected. There really is little sense in trying to decode invalid
  // base64, because it's probably some kind of attack. 
  
  // If the length is not a multiple of 4, it's invalid base64.
  // Here, the empty string will be valid base64 because it represents empty data.
  if( len % 4 )  return 0;
    
  // LOOKING FOR BAD CHARACTERS
  int i;
  for( i = 0 ; i < len - 2 ; i++ )
  {
    if( !nibble_isbase64ValidChr( ascii[i] ) ) 
    {
      printf( "ERROR in nibble_base64integrity at chr %d [%c]. String is NOT valid base64.\n", i, ascii[i] ) ;
      return 0 ;
    }
  }
  
  // Only last 2 can be '='
  // Check 2nd last:
  if( ascii[i]=='=' )
  {
    // If the 2nd last is = the last MUST be = too
    if( ascii[i+1] != '=' )
    {
      printf( "ERROR in nibble_base64integrity at chr %d.\n"
      "If the 2nd last chr is '=' then the last chr must be '=' too.\n "
      "String is NOT valid base64.", i ) ;
      return 0 ;
    }
  }
  else if( !nibble_isbase64ValidChr( ascii[i] ) )  // not = or valid base64
  {
    // 2nd last was invalid and not '='
    printf( "ERROR in nibble_base64integrity at chr %d (2nd last chr). String is NOT valid base64.\n", i ) ;
    return 0 ;
  }
  
  // check last
  
  i++ ;
  if( ascii[i]!='=' && !nibble_isbase64ValidChr( ascii[i] ) )
  {
    printf( "ERROR in nibble_base64integrity at chr %d (last chr). String is NOT valid base64.\n", i ) ;
    return 0 ;    
  }
  
  // Otherwise if get here, b64 string was valid.
  
  return 1 ;
}
