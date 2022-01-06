program test
  use qmckl
  implicit none
  
  integer, parameter :: amax=100
  integer    :: m , n , k
  integer(8) :: m8, n8, k8
  
  ! Matrices
  double precision, allocatable :: A(:,:), B(:,:), C0(:,:), C1(:,:)
  integer                       :: LDA, LDB, LDC0, LDC1
  
  ! C Pointers as int64
  integer(8) :: context
  integer(8) :: A_tile, B_tile
  
  
  
  ! Create context for qmckl. It contains the tiling parameters, and the
  ! list of allocated pointers for cleaning
  context = qmckl_context_create()
  
  ! Allocate matrices once for all
  allocate(A(amax,amax), B(amax,amax), C0(amax,amax), C1(amax,amax))
  LDA  = size( A,1)
  LDB  = size( B,1)
  LDC0 = size(C0,1)
  LDC1 = size(C1,1)
  
  ! Create random matrices A and B
  call random_number(A)
  call random_number(B)
  
  ! For all (m,n,k) in (1..amax)^3, compute C0 = A.B using MKL and
  ! C1 = A.B using qmckl_dgemm, and compare the results
  do m=1,amax
     m8 = m
     do n=1,amax
        n8 = n

        rc = qmckl_tile(qmckl_context, 'C', m8, n8, C1, LDC1, C_tile)
        if (rc /= QMCKL_SUCCESS) then
           print *, m,n
           print *, 'Failed tiling of C1'
           call exit(-1)
        end if
         
      
        do k=1,amax
           k8 = k
         
           rc = qmckl_tile(qmckl_context, 'A', m8, k8, A, LDA, A_tile)
           if (rc /= QMCKL_SUCCESS) then
              print *, m,n,k
              print *, 'Failed tiling of A'
              call exit(-1)
           end if
           
           rc = qmckl_tile(qmckl_context, 'B', k8, n8, B, LDB, B_tile)
           if (rc /= QMCKL_SUCCESS) then
              print *, m,n,k
              print *, 'Failed tiling of B'
              call exit(-1)
           end if
           
           rc = qmckl_dgemm_tiled_NN(qmckl_context, 1.d0, A_tiled, B_tiled, 0.d0, C_tiled)
           if (rc /= QMCKL_SUCCESS) then
              print *, m,n,k
              print *, 'Failed tiled dgemm'
              call exit(-1)
           end if
           
           rc = qmckl_untile(qmckl_context, C_tiled, C1, LDC1)
           if (rc /= QMCKL_SUCCESS) then
              print *, m,n,k
              print *, 'Failed untiling of C'
              call exit(-1)
           end if
           
           ! Compare results
           call dgemm('N','N', m,n,k, 1.d0, A, LDA, B, LDB, 0.d0, C0, LDC0)

           norm1 = 0.d0
           norm2 = 0.d0
           do j=1,n
              do i=1,m
                 norm1 = norm1 + (C0(i,j) - C1(i,j))**2
                 norm2 = norm2 + C0(i,j)**2
              end do
           end do

           if (dsqrt(norm1/norm2) > 1.d-14) then
              print *, m, n, k
              print *, dsqrt(norm1), dsqrt(norm2), dsqrt(norm1/norm2)
              call exit(-1)
           end if

        end do
     end do
  end do

end program test
