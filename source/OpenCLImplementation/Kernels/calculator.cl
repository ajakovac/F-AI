__kernel void scalarMultHelp(global double* buffer, global double* buffer_small, __local double * target) {
  const size_t globalId = get_global_id(0);
  const size_t localId = get_local_id(0);
  target[localId] = buffer[globalId];
  
  barrier(CLK_LOCAL_MEM_FENCE);
  size_t blockSize = get_local_size(0);
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

__kernel void scalarMultHelp2(global double* buffer, global double* buffer_small) {
  const size_t globalId = get_global_id(0);
  buffer[globalId] = buffer_small[globalId];
}

/* output in buffer[0]. Run on n threads. n = 2^l*/
kernel void ScalarMultiply(global double* A, global double* B, global double* buffer,
			   global double* buffer_small, int n) {
  const size_t gid = get_global_id(0);
  buffer[gid] = A[gid] * B[gid];
  if (gid == 0) {
    const int subthread_size = 256;
    
    ndrange_t ndr = ndrange_1D(n, subthread_size);
    
    //while (n >= 256) {
    clk_event_t marker_event01; // o okozza a HIBAT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //clk_event_t marker_event2;
    if (enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndr, 0, NULL, &marker_event01, 
		   ^(local void* target){
		     scalarMultHelp(buffer, buffer_small, (local double*)target);
		       }, subthread_size*sizeof(double) ) != CLK_SUCCESS) {
      buffer[0] = -1;
      release_event(&marker_event01);
      return;
    }
    n = n/subthread_size;
    ndr = ndrange_1D(n, n);//subthread_size);
    if (enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, 
		     ndr, 1, &marker_event01, NULL, 
		       ^{scalarMultHelp2(buffer, buffer_small); }) != CLK_SUCCESS) {
      release_event(&marker_event01);
      buffer[0] = -2;
      return;
      }
      /*
      enqueue_marker(get_default_queue(), 1, &marker_event2, NULL);
      */
    release_event(&marker_event01);
     //release_event(&marker_event2);

  /*
    if (n != 1) {
      ndr = ndrange_1D(n, n);
      enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndr, 0, NULL, NULL, 
		     ^(local void* target){
		       scalarMultHelp(buffer, buffer_small, (local double*)target);
		     }, n*sizeof(double));
    }
  */
      //}
  }
}
