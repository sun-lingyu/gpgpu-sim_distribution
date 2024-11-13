// author: Mahmoud Khairy, (Purdue Univ)
// email: abdallm@purdue.edu

#include <math.h>
#include <string.h>
#include "../abstract_hardware_model.h"
#include "gpu-cache.h"

unsigned ipoly_hash_function(new_addr_type higher_bits, unsigned index,
                             unsigned bank_set_num) {
  /*
   * Set Indexing function from "Pseudo-randomly interleaved memory."
   * Rau, B. R et al.
   * ISCA 1991
   * http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=348DEA37A3E440473B3C075EAABC63B6?doi=10.1.1.12.7149&rep=rep1&type=pdf
   *
   * equations are corresponding to IPOLY(37) and are adopted from:
   * "Sacat: streaming-aware conflict-avoiding thrashing-resistant gpgpu
   * cache management scheme." Khairy et al. IEEE TPDS 2017.
   *
   * equations for 16 banks are corresponding to IPOLY(19)
   * equations for 32 banks are corresponding to IPOLY(37)
   * equations for 64 banks are corresponding to IPOLY(67)
   * equations for 128 banks are corresponding to IPOLY(131)
   * equations for 256 banks are corresponding to IPOLY(283)
   * To see all the IPOLY equations for all the degrees, see
   * http://wireless-systems.ece.gatech.edu/6604/handouts/Peterson's%20Table.pdf
   * or https://www.ece.unb.ca/tervo/ee4253/polyprime.shtml
   *
   * We generate these equations using GF(2) arithmetic:
   * http://www.ee.unb.ca/cgi-bin/tervo/calc.pl?num=&den=&f=d&e=1&m=1
   *
   * We go through all the strides 128 (10000000), 256 (100000000),...  and
   * do modular arithmetic in GF(2) Then, we create the H-matrix and group
   * each bit together, for more info read the ISCA 1991 paper
   *
   * IPOLY hashing guarantees conflict-free for all 2^n strides which widely
   * exit in GPGPU applications and also show good performance for other
   * strides.
   */
  if (bank_set_num == 8) {
    std::bitset<64> a(higher_bits);
    std::bitset<3> b(index);
    std::bitset<3> new_index(index);

    new_index[0] = a[11] ^ a[10] ^ a[9] ^ a[7] ^ a[4] ^ a[3] ^ a[2] ^ a[0] ^ b[0];
    new_index[1] = a[12] ^ a[9] ^ a[8] ^ a[7] ^ a[5] ^ a[2] ^ a[1] ^ a[0] ^ b[1];
    new_index[2] = a[13] ^ a[10] ^ a[9] ^ a[8] ^ a[6] ^ a[3] ^ a[2] ^ a[1] ^ b[2];

    return new_index.to_ulong();

  } else if (bank_set_num == 16) {
    std::bitset<64> a(higher_bits);
    std::bitset<4> b(index);
    std::bitset<4> new_index(index);

    new_index[0] =
        a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[6] ^ a[4] ^ a[3] ^ a[0] ^ b[0];
    new_index[1] =
        a[12] ^ a[8] ^ a[7] ^ a[6] ^ a[5] ^ a[3] ^ a[1] ^ a[0] ^ b[1];
    new_index[2] = a[9] ^ a[8] ^ a[7] ^ a[6] ^ a[4] ^ a[2] ^ a[1] ^ b[2];
    new_index[3] = a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[5] ^ a[3] ^ a[2] ^ b[3];

    return new_index.to_ulong();

  } else if (bank_set_num == 32) {
    std::bitset<64> a(higher_bits);
    std::bitset<5> b(index);
    std::bitset<5> new_index(index);

    new_index[0] =
        a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[9] ^ a[6] ^ a[5] ^ a[3] ^ a[0] ^ b[0];
    new_index[1] = a[14] ^ a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[7] ^ a[6] ^ a[4] ^
                   a[1] ^ b[1];
    new_index[2] =
        a[14] ^ a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[6] ^ a[3] ^ a[2] ^ a[0] ^ b[2];
    new_index[3] =
        a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[4] ^ a[3] ^ a[1] ^ b[3];
    new_index[4] =
        a[12] ^ a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[5] ^ a[4] ^ a[2] ^ b[4];
    return new_index.to_ulong();

  } else if (bank_set_num == 64) {
    std::bitset<64> a(higher_bits);
    std::bitset<6> b(index);
    std::bitset<6> new_index(index);

    new_index[0] = a[18] ^ a[17] ^ a[16] ^ a[15] ^ a[12] ^ a[10] ^ a[6] ^ a[5] ^
                   a[0] ^ b[0];
    new_index[1] = a[15] ^ a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[7] ^ a[5] ^ a[1] ^
                   a[0] ^ b[1];
    new_index[2] = a[16] ^ a[14] ^ a[13] ^ a[12] ^ a[11] ^ a[8] ^ a[6] ^ a[2] ^
                   a[1] ^ b[2];
    new_index[3] = a[17] ^ a[15] ^ a[14] ^ a[13] ^ a[12] ^ a[9] ^ a[7] ^ a[3] ^
                   a[2] ^ b[3];
    new_index[4] = a[18] ^ a[16] ^ a[15] ^ a[14] ^ a[13] ^ a[10] ^ a[8] ^ a[4] ^
                   a[3] ^ b[4];
    new_index[5] =
        a[17] ^ a[16] ^ a[15] ^ a[14] ^ a[11] ^ a[9] ^ a[5] ^ a[4] ^ b[5];
    return new_index.to_ulong();

  } else if (bank_set_num == 128) {
    std::bitset<64> a(higher_bits);
    std::bitset<7> b(index);
    std::bitset<7> new_index(index);

    new_index[0] = a[21] ^ a[20] ^ a[19] ^ a[18] ^ a[14] ^ a[12] ^ a[7] ^ a[6] ^ a[0] ^ b[0];
    new_index[1] = a[22] ^ a[18] ^ a[15] ^ a[14] ^ a[13] ^ a[12] ^ a[8] ^ a[6] ^ a[1] ^ a[0] ^ b[1];
    new_index[2] = a[19] ^ a[16] ^ a[15] ^ a[14] ^ a[13] ^ a[9] ^ a[7] ^ a[2] ^ a[1] ^ b[2];
    new_index[3] = a[20] ^ a[17] ^ a[16] ^ a[15] ^ a[14] ^ a[10] ^ a[8] ^ a[3] ^ a[2] ^ b[3];
    new_index[4] = a[21] ^ a[18] ^ a[17] ^ a[16] ^ a[15] ^ a[11] ^ a[9] ^ a[4] ^ a[3] ^ b[4];
    new_index[5] = a[22] ^ a[19] ^ a[18] ^ a[17] ^ a[16] ^ a[12] ^ a[10] ^ a[5] ^ a[4] ^ b[5];
    new_index[6] = a[20] ^ a[19] ^ a[18] ^ a[17] ^ a[13] ^ a[11] ^ a[6] ^ a[5] ^ b[6];
    return new_index.to_ulong();

  } else if (bank_set_num == 256) {
    std::bitset<64> a(higher_bits);
    std::bitset<8> b(index);
    std::bitset<8> new_index(index);

    new_index[0] = a[21] ^ a[20] ^ a[19] ^ a[17] ^ a[16] ^ a[13] ^ a[12] ^ a[10] ^ a[7] ^ a[5] ^ a[4] ^ a[0] ^ b[0];
    new_index[1] = a[19] ^ a[18] ^ a[16] ^ a[14] ^ a[12] ^ a[11] ^ a[10] ^ a[8] ^ a[7] ^ a[6] ^ a[4] ^ a[1] ^ a[0] ^ b[1];
    new_index[2] = a[20] ^ a[19] ^ a[17] ^ a[15] ^ a[13] ^ a[12] ^ a[11] ^ a[9] ^ a[8] ^ a[7] ^ a[5] ^ a[2] ^ a[1] ^ b[2];
    new_index[3] = a[19] ^ a[18] ^ a[17] ^ a[14] ^ a[9] ^ a[8] ^ a[7] ^ a[6] ^ a[5] ^ a[4] ^ a[3] ^ a[2] ^ a[0] ^ b[3];
    new_index[4] = a[21] ^ a[18] ^ a[17] ^ a[16] ^ a[15] ^ a[13] ^ a[12] ^ a[9] ^ a[8] ^ a[6] ^ a[3] ^ a[1] ^ a[0] ^ b[4];
    new_index[5] = a[19] ^ a[18] ^ a[17] ^ a[16] ^ a[14] ^ a[13] ^ a[10] ^ a[9] ^ a[7] ^ a[4] ^ a[2] ^ a[1] ^ b[5];
    new_index[6] = a[20] ^ a[19] ^ a[18] ^ a[17] ^ a[15] ^ a[14] ^ a[11] ^ a[10] ^ a[8] ^ a[5] ^ a[3] ^ a[2] ^ b[6];
    new_index[7] = a[21] ^ a[20] ^ a[19] ^ a[18] ^ a[16] ^ a[15] ^ a[12] ^ a[11] ^ a[9] ^ a[6] ^ a[4] ^ a[3] ^ b[7];
    return new_index.to_ulong();
  } else { /* Else incorrect number of channels for the hashing function */
    assert(
        "\nmemory_partition_indexing error: The number of "
        "channels should be "
        "8, 16, 32, 64, 128 or 256 for the hashing IPOLY index function. other banks "
        "numbers are not supported. Generate it by yourself! \n" &&
        0);

    return 0;
  }
}

unsigned bitwise_hash_function(new_addr_type higher_bits, unsigned index,
                               unsigned bank_set_num) {
  return (index) ^ (higher_bits & (bank_set_num - 1));
}

unsigned PAE_hash_function(new_addr_type higher_bits, unsigned index,
                           unsigned bank_set_num) {
  // Page Address Entropy
  // random selected bits from the page and bank bits
  // similar to
  // Liu, Yuxi, et al. "Get Out of the Valley: Power-Efficient Address
  if (bank_set_num == 32) {
    std::bitset<64> a(higher_bits);
    std::bitset<5> b(index);
    std::bitset<5> new_index(index);
    new_index[0] = a[13] ^ a[10] ^ a[9] ^ a[5] ^ a[0] ^ b[3] ^ b[0] ^ b[0];
    new_index[1] = a[12] ^ a[11] ^ a[6] ^ a[1] ^ b[3] ^ b[2] ^ b[1] ^ b[1];
    new_index[2] = a[14] ^ a[9] ^ a[8] ^ a[7] ^ a[2] ^ b[1] ^ b[2];
    new_index[3] = a[11] ^ a[10] ^ a[8] ^ a[3] ^ b[2] ^ b[3] ^ b[3];
    new_index[4] = a[12] ^ a[9] ^ a[8] ^ a[5] ^ a[4] ^ b[1] ^ b[0] ^ b[4];

    return new_index.to_ulong();
  } else {
    assert(0);
    return 0;
  }
}
