__kernel void scalarMult2(global double* A, global double* B, 
                          global double* buffer_small, int N, __local double * target) {
  const size_t gid = get_global_id(0);
  const size_t localId = get_local_id(0);
  if (gid < N) target[localId] = A[gid]*B[gid];
  else target[localId] = 0.0;  // zero padding
  
  barrier(CLK_LOCAL_MEM_FENCE);
  size_t blockSize;
  if (gid + get_local_size(0) >= get_global_size(0) ) blockSize = N%get_local_size(0);
  else blockSize = get_local_size(0);
  size_t halfBlockSize = blockSize / 2;

  while (halfBlockSize>0) {
    if (localId<halfBlockSize) {
      target[localId] += target[localId + halfBlockSize];
      if ((halfBlockSize*2)<blockSize) { // uneven block division
        if (localId==0) { // when localID==0
         target[localId] += target[localId + (blockSize-1)];
         }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    blockSize = halfBlockSize; 
    halfBlockSize = blockSize / 2;
  }
  if (localId==0) {
    buffer_small[get_group_id(0)] = target[0];
  }

}

__kernel void sumTo(global double* buffer, global double* buffer_small, int N, __local double * target) {
  const size_t gid = get_global_id(0);
  const size_t localId = get_local_id(0);
  if (gid < N) target[localId] = buffer[gid];
  else target[localId] = 0.0;  // zero padding
  
  barrier(CLK_LOCAL_MEM_FENCE);
  size_t blockSize;
  if (gid + get_local_size(0) >= get_global_size(0) ) blockSize = N%get_local_size(0);
  else blockSize = get_local_size(0);
  size_t halfBlockSize = blockSize / 2;

  while (halfBlockSize>0) {
    if (localId<halfBlockSize) {
      target[localId] += target[localId + halfBlockSize];
      if ((halfBlockSize*2)<blockSize) { // uneven block division
        if (localId==0) { // when localID==0
	       target[localId] += target[localId + (blockSize-1)];
	       }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    blockSize = halfBlockSize; 
    halfBlockSize = blockSize / 2;
  }
  if (localId==0) {
    buffer_small[get_group_id(0)] = target[0];
  }

}

__kernel void copyTo(global double* buffer, global double* buffer1, int n) {
  const size_t gid = get_global_id(0);
  if (gid < n) buffer1[gid] = buffer[gid];
}

kernel void pointwiseMultiply(__global double* A, global double* B, global double* buffer, int n, int k) {
  const size_t gid = get_global_id(0);
  if (gid < n) buffer[gid] = A[gid]* B[gid];
}

kernel void empty() {}