subroutine print_matrix(A, N, M)
  implicit none
  integer,intent(in) :: M, N
  double precision,intent(in) :: A(N,M)
  integer :: i, j
  print *, "Printing Matrix", N, M
  do j = 1, M
  do i = 1, N
     print *,A(i,j)
  end do
  end do
end subroutine print_matrix


program test
  use qmckl_dgemm
  implicit none
  
  integer, parameter :: amax=20
  integer    :: m , n , k, i, j, ii, jj
  double precision :: norm1, norm2
  integer(8) :: m8, n8, k8
  integer(qmckl_exit_code) :: res
  
  ! Matrices
  double precision, allocatable :: A(:,:), B(:,:), C0(:,:), C1(:,:)
  integer(8)                    :: LDA, LDB, LDC0, LDC1
  
  ! C Pointers as int64
  integer(8) :: context
  !integer(8) :: A_tile, B_tile, C_tile
  !integer(8) :: A_tile, B_tile, C_tile
  double precision, allocatable :: A_tile(:), B_tile(:), C_tile(:)
  integer(8) :: rc 
  
  
  ! Create context for qmckl. It contains the tiling parameters, and the
  ! list of allocated pointers for cleaning
  
  ! For all (m,n,k) in (1..amax)^3, compute C0 = A.B using MKL and
  ! C1 = A.B using qmckl_dgemm, and compare the results
  do m=1,amax
     m8 = m
     do n=1,amax
        n8 = n

        do k=1,amax
           k8 = k

           context = qmckl_context_create()

           ! Allocate matrices once for all
           allocate(A(k8,m8), B(n8,k8), C0(m8,n8), C1(n8,m8))
           LDA  = size( A,1)
           LDB  = size( B,1)
           LDC0 = size(C0,1)
           LDC1 = size(C1,1)
  
           ! Create random matrices A and B
           call random_number(A)
           call random_number(B)


           C0 = 0.0d0
           C1 = 0.0d0

           rc = qmckl_init_pack(context, 'C', m8, n8, k8)
           rc = qmckl_pack_matrix(context, 'C', m8, n8, C1, LDC1)
           if (rc /= QMCKL_SUCCESS) then
              print *, m,n
              print *, 'Failed tiling of C1'
              call exit(-1)
           end if

         
           rc = qmckl_init_pack(context, 'A', m8, k8, k8)
           rc = qmckl_pack_matrix(context, 'A', m8, k8, A, LDA)
           if (rc /= QMCKL_SUCCESS) then
              print *, m,n,k
              print *, 'Failed tiling of A'
              call exit(-1)
           end if
           
           rc = qmckl_init_pack(context, 'B', k8, n8, k8)
           rc = qmckl_pack_matrix(context, 'B', k8, n8, B, LDB)
           if (rc /= QMCKL_SUCCESS) then
              print *, m,n,k
              print *, 'Failed tiling of B'
              call exit(-1)
           end if
           
           
           rc = qmckl_dgemm_tiled_avx2_nn(context, A, LDA, B, LDB, C1, LDC1)

           if (rc /= QMCKL_SUCCESS) then
              print *, m,n,k
              print *, 'Failed tiled dgemm'
              call exit(-1)
           end if
           
           rc = qmckl_unpack_matrix(context, C1, m8, n8)
           if (rc /= QMCKL_SUCCESS) then
              print *, m,n,k
              print *, 'Failed untiling of C'
              call exit(-1)
           end if

           ! Compare results
           call dgemm('T','T', m,n,k, 1.d0, A, LDA, B, LDB, 0.d0, C0, LDC0)

           norm1 = 0.d0
           norm2 = 0.d0
           do j=1,n
              do i=1,m
                 norm1 = norm1 + (C0(i,j) - C1(j,i))**2
                 norm2 = norm2 + C0(i,j)**2
              end do
           end do

           C0 = 0.0d0

           if (dsqrt(norm1/norm2) > 1.d-14) then
              print *, m, n, k
              print *, dsqrt(norm1), dsqrt(norm2), dsqrt(norm1/norm2)
              print *, 'Failed DGEMM'
              call exit(-1)
           end if

           deallocate(A)
           deallocate(B)
           deallocate(C0)
           deallocate(C1)
           res = qmckl_context_destroy(context)

        end do
     end do
  end do

end program test
