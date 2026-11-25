	.file	"vector_search_core.f90"
	.text
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC1:
	.string	"Query and database vectors must have equal dimensions."
	.text
	.p2align 4
	.globl	__fnlp_vector_search_core_MOD_ip_database_openmp
	.type	__fnlp_vector_search_core_MOD_ip_database_openmp, @function
__fnlp_vector_search_core_MOD_ip_database_openmp:
.LFB4:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movl	$1, %r11d
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	movq	%rdx, %r13
	pushq	%r12
	pushq	%rbx
	andq	$-64, %rsp
	subq	$64, %rsp
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	40(%rsi), %r9
	movq	56(%rsi), %rdx
	movq	80(%rsi), %rcx
	movq	40(%rdi), %r10
	testq	%r9, %r9
	movq	56(%rdi), %r12
	movq	(%rsi), %r8
	cmove	%r11, %r9
	subq	48(%rsi), %rdx
	subq	72(%rsi), %rcx
	incq	%rdx
	movq	64(%rsi), %r14
	testq	%r10, %r10
	movq	(%rdi), %rax
	leaq	1(%rcx), %rsi
	cmove	%r11, %r10
	subq	48(%rdi), %r12
	xorl	%edi, %edi
	testq	%rsi, %rsi
	cmovns	%rsi, %rdi
	xorl	%esi, %esi
	movq	%rdi, 56(%rsp)
	leaq	1(%r12), %rdi
	testq	%rdi, %rdi
	cmovs	%rsi, %rdi
	testq	%rdx, %rdx
	cmovs	%rsi, %rdx
	cmpq	%rdi, %rdx
	jne	.L36
	testq	%rcx, %rcx
	js	.L33
	cmpq	$1, %r9
	jne	.L24
	cmpq	$1, %r10
	jne	.L24
	testq	%r12, %r12
	movl	$1, %r10d
	cmovns	%rdi, %r11
	movq	%r11, %rsi
	movq	%r11, %rbx
	movq	%r11, %r9
	movq	%r11, 48(%rsp)
	andl	$15, %esi
	andq	$-16, %rbx
	shrq	$4, %r9
	movq	%rsi, 40(%rsp)
	salq	$6, %r9
	movq	%rbx, 32(%rsp)
	incq	%rbx
	movq	%rbx, 24(%rsp)
	xorl	%ebx, %ebx
	.p2align 4,,10
	.p2align 3
.L20:
	movl	$0x00000000, -4(%r13,%r10,4)
	leaq	0(,%rbx,4), %rsi
	testq	%r12, %r12
	js	.L12
	cmpq	$14, %r12
	jle	.L23
	leaq	(%r8,%rsi), %rcx
	xorl	%edx, %edx
	vxorps	%xmm0, %xmm0, %xmm0
	.p2align 4,,10
	.p2align 3
.L15:
	vmovups	(%rax,%rdx), %zmm5
	vmulps	(%rcx,%rdx), %zmm5, %zmm2
	addq	$64, %rdx
	vshufps	$85, %xmm2, %xmm2, %xmm1
	vshufps	$255, %xmm2, %xmm2, %xmm4
	valignd	$7, %ymm2, %ymm2, %ymm3
	vaddss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	vunpckhps	%xmm2, %xmm2, %xmm1
	vaddss	%xmm1, %xmm0, %xmm0
	valignd	$6, %ymm2, %ymm2, %ymm1
	vaddss	%xmm4, %xmm0, %xmm0
	vextractf32x4	$1, %ymm2, %xmm4
	vaddss	%xmm4, %xmm0, %xmm0
	valignd	$5, %ymm2, %ymm2, %ymm4
	vaddss	%xmm4, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	vextractf32x8	$0x1, %zmm2, %ymm1
	vshufps	$255, %xmm1, %xmm1, %xmm2
	vaddss	%xmm3, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm3
	vaddss	%xmm1, %xmm0, %xmm0
	vaddss	%xmm3, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm3
	vaddss	%xmm3, %xmm0, %xmm0
	vaddss	%xmm2, %xmm0, %xmm0
	vextractf32x4	$1, %ymm1, %xmm2
	vaddss	%xmm2, %xmm0, %xmm0
	valignd	$5, %ymm1, %ymm1, %ymm2
	vaddss	%xmm2, %xmm0, %xmm0
	valignd	$6, %ymm1, %ymm1, %ymm2
	valignd	$7, %ymm1, %ymm1, %ymm1
	vaddss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	cmpq	%rdx, %r9
	jne	.L15
	cmpq	$0, 40(%rsp)
	je	.L16
	movq	32(%rsp), %r11
	movq	24(%rsp), %rdx
.L13:
	movq	48(%rsp), %rcx
	subq	%r11, %rcx
	leaq	-1(%rcx), %r15
	cmpq	$6, %r15
	jbe	.L17
	vmovups	(%rax,%r11,4), %ymm1
	leaq	(%rbx,%r11), %r15
	movq	%rcx, %r11
	andq	$-8, %r11
	vmulps	(%r8,%r15,4), %ymm1, %ymm1
	addq	%r11, %rdx
	andl	$7, %ecx
	vaddss	%xmm1, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm3
	vshufps	$255, %xmm1, %xmm1, %xmm2
	vaddss	%xmm3, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm3
	vaddss	%xmm3, %xmm0, %xmm0
	vaddss	%xmm2, %xmm0, %xmm0
	vextractf32x4	$1, %ymm1, %xmm2
	vaddss	%xmm2, %xmm0, %xmm0
	valignd	$5, %ymm1, %ymm1, %ymm2
	vaddss	%xmm2, %xmm0, %xmm0
	valignd	$6, %ymm1, %ymm1, %ymm2
	valignd	$7, %ymm1, %ymm1, %ymm1
	vaddss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	je	.L16
.L17:
	leaq	-1(%rdx), %r11
	leaq	0(,%r11,4), %rcx
	leaq	(%r8,%rcx), %r15
	vmovss	(%r15,%rsi), %xmm7
	vfmadd231ss	(%rax,%r11,4), %xmm7, %xmm0
	cmpq	%rdx, %rdi
	jle	.L16
	leaq	4(%r8,%rcx), %r11
	vmovss	(%r11,%rsi), %xmm7
	leaq	2(%rdx), %r11
	vfmadd231ss	4(%rax,%rcx), %xmm7, %xmm0
	cmpq	%r11, %rdi
	jl	.L16
	vmovss	8(%rax,%rcx), %xmm7
	addq	%r8, %rsi
	leaq	3(%rdx), %r11
	vfmadd231ss	8(%rcx,%rsi), %xmm7, %xmm0
	cmpq	%r11, %rdi
	jl	.L16
	vmovss	12(%rax,%rcx), %xmm7
	leaq	4(%rdx), %r11
	vfmadd231ss	12(%rcx,%rsi), %xmm7, %xmm0
	cmpq	%r11, %rdi
	jl	.L16
	vmovss	16(%rax,%rcx), %xmm6
	leaq	5(%rdx), %r11
	vfmadd231ss	16(%rcx,%rsi), %xmm6, %xmm0
	cmpq	%r11, %rdi
	jl	.L16
	vmovss	20(%rax,%rcx), %xmm6
	addq	$6, %rdx
	vfmadd231ss	20(%rcx,%rsi), %xmm6, %xmm0
	cmpq	%rdx, %rdi
	jl	.L16
	vmovss	24(%rax,%rcx), %xmm7
	vfmadd231ss	24(%rcx,%rsi), %xmm7, %xmm0
.L16:
	vmovss	%xmm0, -4(%r13,%r10,4)
.L12:
	incq	%r10
	addq	%r14, %rbx
	cmpq	%r10, 56(%rsp)
	jge	.L20
	vzeroupper
.L33:
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L23:
	.cfi_restore_state
	xorl	%r11d, %r11d
	vxorps	%xmm0, %xmm0, %xmm0
	movl	$1, %edx
	jmp	.L13
.L24:
	movq	56(%rsp), %rbx
	salq	$2, %r14
	salq	$2, %r10
	movl	$1, %r11d
	salq	$2, %r9
	.p2align 4,,10
	.p2align 3
.L10:
	movl	$0x00000000, -4(%r13,%r11,4)
	testq	%r12, %r12
	js	.L8
	movq	%rax, %rsi
	movq	%r8, %rcx
	vxorps	%xmm0, %xmm0, %xmm0
	movl	$1, %edx
	.p2align 4,,10
	.p2align 3
.L9:
	vmovss	(%rsi), %xmm6
	incq	%rdx
	addq	%r10, %rsi
	vfmadd231ss	(%rcx), %xmm6, %xmm0
	addq	%r9, %rcx
	cmpq	%rdx, %rdi
	jge	.L9
	vmovss	%xmm0, -4(%r13,%r11,4)
.L8:
	incq	%r11
	addq	%r14, %r8
	cmpq	%r11, %rbx
	jge	.L10
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L36:
	.cfi_restore_state
	xorl	%edx, %edx
	movl	$54, %esi
	leaq	.LC1(%rip), %rdi
	call	_gfortran_error_stop_string@PLT
	.cfi_endproc
.LFE4:
	.size	__fnlp_vector_search_core_MOD_ip_database_openmp, .-__fnlp_vector_search_core_MOD_ip_database_openmp
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC2:
	.string	"k must be at least 1."
	.section	.rodata.str1.8
	.align 8
.LC3:
	.string	"k cannot exceed the database size."
	.align 8
.LC4:
	.string	"Integer overflow when calculating the amount of memory to allocate"
	.section	.rodata.str1.1
.LC5:
	.string	"Error allocating %lu bytes"
	.section	.rodata.str1.8
	.align 8
.LC6:
	.string	"In file 'src/native/cpu/vector_search/vector_search_core.f90', around line 360"
	.text
	.p2align 4
	.globl	__fnlp_vector_search_core_MOD_ip_search_openmp
	.type	__fnlp_vector_search_core_MOD_ip_search_openmp, @function
__fnlp_vector_search_core_MOD_ip_search_openmp:
.LFB0:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	movq	%rsi, %rax
	movl	$1, %r10d
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	movq	%rdx, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$296, %rsp
	.cfi_def_cfa_offset 352
	movq	40(%rsi), %rdx
	movq	56(%rsi), %rbx
	movq	%rcx, 16(%rsp)
	movq	80(%rax), %rcx
	movq	(%rsi), %r12
	movq	%r8, 24(%rsp)
	testq	%rdx, %rdx
	movq	64(%rax), %r13
	movq	56(%rdi), %r14
	cmove	%r10, %rdx
	subq	72(%rax), %rcx
	subq	48(%rsi), %rbx
	leaq	1(%rbx), %rsi
	leaq	1(%rcx), %rbx
	movq	%rcx, %rax
	movq	40(%rdi), %rcx
	movslq	0(%rbp), %r15
	movq	%rsi, (%rsp)
	testq	%rcx, %rcx
	cmove	%r10, %rcx
	subq	48(%rdi), %r14
	xorl	%r11d, %r11d
	testq	%rbx, %rbx
	leaq	1(%r14), %rsi
	movq	(%rdi), %r10
	cmovns	%rbx, %r11
	movq	%rsi, 8(%rsp)
	testl	%r15d, %r15d
	jle	.L49
	movq	%r11, %r14
	cmpq	%r11, %r15
	jg	.L50
	movabsq	$3302829850624, %rsi
	movabsq	$4611686018427387903, %rdi
	movq	$4, 80(%rsp)
	shrq	$63, %rax
	movq	%rsi, 88(%rsp)
	cmpq	%rdi, %rbx
	jg	.L42
	testl	%eax, %eax
	jne	.L47
	leaq	0(,%r11,4), %rax
	movq	%rax, 56(%rsp)
	movq	%rax, %rdi
.L43:
	movq	%r10, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	%rdx, 32(%rsp)
	call	malloc@PLT
	movq	32(%rsp), %rdx
	movq	40(%rsp), %rcx
	testq	%rax, %rax
	movq	%rax, 64(%rsp)
	movq	48(%rsp), %r10
	movq	%rax, %r15
	je	.L51
	movq	8(%rsp), %rax
	vmovdqa	.LC7(%rip), %xmm0
	movq	%rdx, 232(%rsp)
	leaq	192(%rsp), %rsi
	leaq	128(%rsp), %rdi
	movq	%r15, %rdx
	movq	%r14, 120(%rsp)
	movq	%rax, 184(%rsp)
	movq	(%rsp), %rax
	movq	%rcx, 168(%rsp)
	movq	%r10, 128(%rsp)
	movq	%rax, 248(%rsp)
	movq	%rbx, 272(%rsp)
	movq	%r13, 256(%rsp)
	movq	%r12, 192(%rsp)
	vmovdqa	%xmm0, 96(%rsp)
	movq	$1, 112(%rsp)
	movq	$-1, 72(%rsp)
	movq	$1, 176(%rsp)
	movq	$1, 240(%rsp)
	movq	$1, 264(%rsp)
	call	__fnlp_vector_search_core_MOD_ip_database_openmp
	movq	24(%rsp), %rcx
	movq	16(%rsp), %rdx
	leaq	64(%rsp), %rdi
	movq	%rbp, %rsi
	call	__fnlp_topk_core_MOD_topk_largest@PLT
	addq	$296, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	movq	%r15, %rdi
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmp	free@PLT
	.p2align 4,,10
	.p2align 3
.L47:
	.cfi_restore_state
	movq	$0, 56(%rsp)
	movl	$1, %edi
	jmp	.L43
.L51:
	movq	56(%rsp), %rdx
	leaq	.LC5(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	xorl	%eax, %eax
	call	_gfortran_os_error_at@PLT
.L42:
	leaq	.LC4(%rip), %rdi
	xorl	%eax, %eax
	call	_gfortran_runtime_error@PLT
.L50:
	xorl	%edx, %edx
	movl	$34, %esi
	leaq	.LC3(%rip), %rdi
	call	_gfortran_error_stop_string@PLT
.L49:
	xorl	%edx, %edx
	movl	$21, %esi
	leaq	.LC2(%rip), %rdi
	call	_gfortran_error_stop_string@PLT
	.cfi_endproc
.LFE0:
	.size	__fnlp_vector_search_core_MOD_ip_search_openmp, .-__fnlp_vector_search_core_MOD_ip_search_openmp
	.section	.rodata.str1.8
	.align 8
.LC8:
	.string	"In file 'src/native/cpu/vector_search/vector_search_core.f90', around line 318"
	.text
	.p2align 4
	.globl	__fnlp_vector_search_core_MOD_ip_search_serial
	.type	__fnlp_vector_search_core_MOD_ip_search_serial, @function
__fnlp_vector_search_core_MOD_ip_search_serial:
.LFB1:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	movq	%rsi, %rax
	movl	$1, %r10d
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	movq	%rdx, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$296, %rsp
	.cfi_def_cfa_offset 352
	movq	40(%rsi), %rdx
	movq	56(%rsi), %rbx
	movq	%rcx, 16(%rsp)
	movq	80(%rax), %rcx
	movq	(%rsi), %r12
	movq	%r8, 24(%rsp)
	testq	%rdx, %rdx
	movq	64(%rax), %r13
	movq	56(%rdi), %r14
	cmove	%r10, %rdx
	subq	72(%rax), %rcx
	subq	48(%rsi), %rbx
	leaq	1(%rbx), %rsi
	leaq	1(%rcx), %rbx
	movq	%rcx, %rax
	movq	40(%rdi), %rcx
	movslq	0(%rbp), %r15
	movq	%rsi, (%rsp)
	testq	%rcx, %rcx
	cmove	%r10, %rcx
	subq	48(%rdi), %r14
	xorl	%r11d, %r11d
	testq	%rbx, %rbx
	leaq	1(%r14), %rsi
	movq	(%rdi), %r10
	cmovns	%rbx, %r11
	movq	%rsi, 8(%rsp)
	testl	%r15d, %r15d
	jle	.L64
	movq	%r11, %r14
	cmpq	%r11, %r15
	jg	.L65
	movabsq	$3302829850624, %rsi
	movabsq	$4611686018427387903, %rdi
	movq	$4, 80(%rsp)
	shrq	$63, %rax
	movq	%rsi, 88(%rsp)
	cmpq	%rdi, %rbx
	jg	.L57
	testl	%eax, %eax
	jne	.L62
	leaq	0(,%r11,4), %rax
	movq	%rax, 56(%rsp)
	movq	%rax, %rdi
.L58:
	movq	%r10, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	%rdx, 32(%rsp)
	call	malloc@PLT
	movq	32(%rsp), %rdx
	movq	40(%rsp), %rcx
	testq	%rax, %rax
	movq	%rax, 64(%rsp)
	movq	48(%rsp), %r10
	movq	%rax, %r15
	je	.L66
	movq	8(%rsp), %rax
	vmovdqa	.LC7(%rip), %xmm0
	movq	%rdx, 232(%rsp)
	leaq	192(%rsp), %rsi
	leaq	128(%rsp), %rdi
	movq	%r15, %rdx
	movq	%r14, 120(%rsp)
	movq	%rax, 184(%rsp)
	movq	(%rsp), %rax
	movq	%rcx, 168(%rsp)
	movq	%r10, 128(%rsp)
	movq	%rax, 248(%rsp)
	movq	%rbx, 272(%rsp)
	movq	%r13, 256(%rsp)
	movq	%r12, 192(%rsp)
	vmovdqa	%xmm0, 96(%rsp)
	movq	$1, 112(%rsp)
	movq	$-1, 72(%rsp)
	movq	$1, 176(%rsp)
	movq	$1, 240(%rsp)
	movq	$1, 264(%rsp)
	call	__fnlp_vector_search_core_MOD_ip_database_openmp
	movq	24(%rsp), %rcx
	movq	16(%rsp), %rdx
	leaq	64(%rsp), %rdi
	movq	%rbp, %rsi
	call	__fnlp_topk_core_MOD_topk_largest@PLT
	addq	$296, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	movq	%r15, %rdi
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmp	free@PLT
	.p2align 4,,10
	.p2align 3
.L62:
	.cfi_restore_state
	movq	$0, 56(%rsp)
	movl	$1, %edi
	jmp	.L58
.L66:
	movq	56(%rsp), %rdx
	leaq	.LC5(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	xorl	%eax, %eax
	call	_gfortran_os_error_at@PLT
.L57:
	leaq	.LC4(%rip), %rdi
	xorl	%eax, %eax
	call	_gfortran_runtime_error@PLT
.L65:
	xorl	%edx, %edx
	movl	$34, %esi
	leaq	.LC3(%rip), %rdi
	call	_gfortran_error_stop_string@PLT
.L64:
	xorl	%edx, %edx
	movl	$21, %esi
	leaq	.LC2(%rip), %rdi
	call	_gfortran_error_stop_string@PLT
	.cfi_endproc
.LFE1:
	.size	__fnlp_vector_search_core_MOD_ip_search_serial, .-__fnlp_vector_search_core_MOD_ip_search_serial
	.p2align 4
	.globl	__fnlp_vector_search_core_MOD_l2_database_openmp
	.type	__fnlp_vector_search_core_MOD_l2_database_openmp, @function
__fnlp_vector_search_core_MOD_l2_database_openmp:
.LFB5:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movl	$1, %r11d
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	movq	%rdx, %r13
	pushq	%r12
	pushq	%rbx
	andq	$-64, %rsp
	subq	$64, %rsp
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	40(%rsi), %r9
	movq	56(%rsi), %rdx
	movq	80(%rsi), %rcx
	movq	40(%rdi), %r10
	testq	%r9, %r9
	movq	56(%rdi), %r12
	movq	(%rsi), %r8
	cmove	%r11, %r9
	subq	48(%rsi), %rdx
	subq	72(%rsi), %rcx
	incq	%rdx
	movq	64(%rsi), %r14
	testq	%r10, %r10
	movq	(%rdi), %rax
	leaq	1(%rcx), %rsi
	cmove	%r11, %r10
	subq	48(%rdi), %r12
	xorl	%edi, %edi
	testq	%rsi, %rsi
	cmovns	%rsi, %rdi
	xorl	%esi, %esi
	movq	%rdi, 56(%rsp)
	leaq	1(%r12), %rdi
	testq	%rdi, %rdi
	cmovs	%rsi, %rdi
	testq	%rdx, %rdx
	cmovs	%rsi, %rdx
	cmpq	%rdi, %rdx
	jne	.L101
	testq	%rcx, %rcx
	js	.L99
	cmpq	$1, %r9
	jne	.L90
	cmpq	$1, %r10
	jne	.L90
	testq	%r12, %r12
	movl	$1, %r10d
	cmovns	%rdi, %r11
	movq	%r11, %rsi
	movq	%r11, %rbx
	movq	%r11, %r9
	movq	%r11, 48(%rsp)
	andl	$15, %esi
	andq	$-16, %rbx
	shrq	$4, %r9
	movq	%rsi, 40(%rsp)
	salq	$6, %r9
	movq	%rbx, 32(%rsp)
	incq	%rbx
	movq	%rbx, 24(%rsp)
	xorl	%ebx, %ebx
	.p2align 4,,10
	.p2align 3
.L86:
	movl	$0x00000000, -4(%r13,%r10,4)
	leaq	0(,%rbx,4), %rsi
	testq	%r12, %r12
	js	.L78
	cmpq	$14, %r12
	jle	.L89
	leaq	(%r8,%rsi), %rcx
	xorl	%edx, %edx
	vxorps	%xmm0, %xmm0, %xmm0
	.p2align 4,,10
	.p2align 3
.L81:
	vmovups	(%rax,%rdx), %zmm5
	vsubps	(%rcx,%rdx), %zmm5, %zmm1
	addq	$64, %rdx
	vmulps	%zmm1, %zmm1, %zmm2
	vshufps	$85, %xmm2, %xmm2, %xmm1
	vshufps	$255, %xmm2, %xmm2, %xmm4
	valignd	$7, %ymm2, %ymm2, %ymm3
	vaddss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	vunpckhps	%xmm2, %xmm2, %xmm1
	vaddss	%xmm1, %xmm0, %xmm0
	valignd	$6, %ymm2, %ymm2, %ymm1
	vaddss	%xmm4, %xmm0, %xmm0
	vextractf32x4	$1, %ymm2, %xmm4
	vaddss	%xmm4, %xmm0, %xmm0
	valignd	$5, %ymm2, %ymm2, %ymm4
	vaddss	%xmm4, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	vextractf32x8	$0x1, %zmm2, %ymm1
	vshufps	$255, %xmm1, %xmm1, %xmm2
	vaddss	%xmm3, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm3
	vaddss	%xmm1, %xmm0, %xmm0
	vaddss	%xmm3, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm3
	vaddss	%xmm3, %xmm0, %xmm0
	vaddss	%xmm2, %xmm0, %xmm0
	vextractf32x4	$1, %ymm1, %xmm2
	vaddss	%xmm2, %xmm0, %xmm0
	valignd	$5, %ymm1, %ymm1, %ymm2
	vaddss	%xmm2, %xmm0, %xmm0
	valignd	$6, %ymm1, %ymm1, %ymm2
	valignd	$7, %ymm1, %ymm1, %ymm1
	vaddss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	cmpq	%rdx, %r9
	jne	.L81
	cmpq	$0, 40(%rsp)
	je	.L82
	movq	32(%rsp), %rcx
	movq	24(%rsp), %rdx
.L79:
	movq	48(%rsp), %r11
	subq	%rcx, %r11
	leaq	-1(%r11), %r15
	cmpq	$6, %r15
	jbe	.L83
	vmovups	(%rax,%rcx,4), %ymm1
	addq	%rbx, %rcx
	vsubps	(%r8,%rcx,4), %ymm1, %ymm1
	movq	%r11, %rcx
	andq	$-8, %rcx
	addq	%rcx, %rdx
	andl	$7, %r11d
	vmulps	%ymm1, %ymm1, %ymm1
	vaddss	%xmm1, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm3
	vshufps	$255, %xmm1, %xmm1, %xmm2
	vaddss	%xmm3, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm3
	vaddss	%xmm3, %xmm0, %xmm0
	vaddss	%xmm2, %xmm0, %xmm0
	vextractf32x4	$1, %ymm1, %xmm2
	vaddss	%xmm2, %xmm0, %xmm0
	valignd	$5, %ymm1, %ymm1, %ymm2
	vaddss	%xmm2, %xmm0, %xmm0
	valignd	$6, %ymm1, %ymm1, %ymm2
	valignd	$7, %ymm1, %ymm1, %ymm1
	vaddss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	je	.L82
.L83:
	leaq	-1(%rdx), %r11
	vmovss	(%rax,%r11,4), %xmm1
	leaq	0(,%r11,4), %rcx
	leaq	(%r8,%rcx), %r15
	vsubss	(%r15,%rsi), %xmm1, %xmm1
	vfmadd231ss	%xmm1, %xmm1, %xmm0
	cmpq	%rdx, %rdi
	jle	.L82
	vmovss	4(%rax,%rcx), %xmm1
	leaq	4(%r8,%rcx), %r11
	vsubss	(%r11,%rsi), %xmm1, %xmm1
	leaq	2(%rdx), %r11
	vfmadd231ss	%xmm1, %xmm1, %xmm0
	cmpq	%r11, %rdi
	jl	.L82
	vmovss	8(%rax,%rcx), %xmm1
	addq	%r8, %rsi
	leaq	3(%rdx), %r11
	vsubss	8(%rcx,%rsi), %xmm1, %xmm1
	vfmadd231ss	%xmm1, %xmm1, %xmm0
	cmpq	%r11, %rdi
	jl	.L82
	vmovss	12(%rax,%rcx), %xmm1
	leaq	4(%rdx), %r11
	vsubss	12(%rcx,%rsi), %xmm1, %xmm1
	vfmadd231ss	%xmm1, %xmm1, %xmm0
	cmpq	%r11, %rdi
	jl	.L82
	vmovss	16(%rax,%rcx), %xmm1
	leaq	5(%rdx), %r11
	vsubss	16(%rcx,%rsi), %xmm1, %xmm1
	vfmadd231ss	%xmm1, %xmm1, %xmm0
	cmpq	%r11, %rdi
	jl	.L82
	vmovss	20(%rax,%rcx), %xmm1
	addq	$6, %rdx
	vsubss	20(%rcx,%rsi), %xmm1, %xmm1
	vfmadd231ss	%xmm1, %xmm1, %xmm0
	cmpq	%rdx, %rdi
	jl	.L82
	vmovss	24(%rax,%rcx), %xmm1
	vsubss	24(%rcx,%rsi), %xmm1, %xmm1
	vfmadd231ss	%xmm1, %xmm1, %xmm0
.L82:
	vmovss	%xmm0, -4(%r13,%r10,4)
.L78:
	incq	%r10
	addq	%r14, %rbx
	cmpq	%r10, 56(%rsp)
	jge	.L86
	vzeroupper
.L99:
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L89:
	.cfi_restore_state
	xorl	%ecx, %ecx
	vxorps	%xmm0, %xmm0, %xmm0
	movl	$1, %edx
	jmp	.L79
.L90:
	movq	56(%rsp), %rbx
	salq	$2, %r14
	salq	$2, %r10
	movl	$1, %r11d
	salq	$2, %r9
	.p2align 4,,10
	.p2align 3
.L76:
	movl	$0x00000000, -4(%r13,%r11,4)
	testq	%r12, %r12
	js	.L74
	movq	%rax, %rsi
	movq	%r8, %rcx
	vxorps	%xmm1, %xmm1, %xmm1
	movl	$1, %edx
	.p2align 4,,10
	.p2align 3
.L75:
	vmovss	(%rsi), %xmm0
	incq	%rdx
	addq	%r10, %rsi
	vsubss	(%rcx), %xmm0, %xmm0
	addq	%r9, %rcx
	vfmadd231ss	%xmm0, %xmm0, %xmm1
	cmpq	%rdx, %rdi
	jge	.L75
	vmovss	%xmm1, -4(%r13,%r11,4)
.L74:
	incq	%r11
	addq	%r14, %r8
	cmpq	%r11, %rbx
	jge	.L76
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L101:
	.cfi_restore_state
	xorl	%edx, %edx
	movl	$54, %esi
	leaq	.LC1(%rip), %rdi
	call	_gfortran_error_stop_string@PLT
	.cfi_endproc
.LFE5:
	.size	__fnlp_vector_search_core_MOD_l2_database_openmp, .-__fnlp_vector_search_core_MOD_l2_database_openmp
	.section	.rodata.str1.8
	.align 8
.LC9:
	.string	"In file 'src/native/cpu/vector_search/vector_search_core.f90', around line 276"
	.text
	.p2align 4
	.globl	__fnlp_vector_search_core_MOD_l2_search_openmp
	.type	__fnlp_vector_search_core_MOD_l2_search_openmp, @function
__fnlp_vector_search_core_MOD_l2_search_openmp:
.LFB2:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	movq	%rsi, %rax
	movl	$1, %r10d
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	movq	%rdx, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$296, %rsp
	.cfi_def_cfa_offset 352
	movq	40(%rsi), %rdx
	movq	56(%rsi), %rbx
	movq	%rcx, 16(%rsp)
	movq	80(%rax), %rcx
	movq	(%rsi), %r12
	movq	%r8, 24(%rsp)
	testq	%rdx, %rdx
	movq	64(%rax), %r13
	movq	56(%rdi), %r14
	cmove	%r10, %rdx
	subq	72(%rax), %rcx
	subq	48(%rsi), %rbx
	leaq	1(%rbx), %rsi
	leaq	1(%rcx), %rbx
	movq	%rcx, %rax
	movq	40(%rdi), %rcx
	movslq	0(%rbp), %r15
	movq	%rsi, (%rsp)
	testq	%rcx, %rcx
	cmove	%r10, %rcx
	subq	48(%rdi), %r14
	xorl	%r11d, %r11d
	testq	%rbx, %rbx
	leaq	1(%r14), %rsi
	movq	(%rdi), %r10
	cmovns	%rbx, %r11
	movq	%rsi, 8(%rsp)
	testl	%r15d, %r15d
	jle	.L114
	movq	%r11, %r14
	cmpq	%r11, %r15
	jg	.L115
	movabsq	$3302829850624, %rsi
	movabsq	$4611686018427387903, %rdi
	movq	$4, 80(%rsp)
	shrq	$63, %rax
	movq	%rsi, 88(%rsp)
	cmpq	%rdi, %rbx
	jg	.L107
	testl	%eax, %eax
	jne	.L112
	leaq	0(,%r11,4), %rax
	movq	%rax, 56(%rsp)
	movq	%rax, %rdi
.L108:
	movq	%r10, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	%rdx, 32(%rsp)
	call	malloc@PLT
	movq	32(%rsp), %rdx
	movq	40(%rsp), %rcx
	testq	%rax, %rax
	movq	%rax, 64(%rsp)
	movq	48(%rsp), %r10
	movq	%rax, %r15
	je	.L116
	movq	8(%rsp), %rax
	vmovdqa	.LC7(%rip), %xmm0
	movq	%rdx, 232(%rsp)
	leaq	192(%rsp), %rsi
	leaq	128(%rsp), %rdi
	movq	%r15, %rdx
	movq	%r14, 120(%rsp)
	movq	%rax, 184(%rsp)
	movq	(%rsp), %rax
	movq	%rcx, 168(%rsp)
	movq	%r10, 128(%rsp)
	movq	%rax, 248(%rsp)
	movq	%rbx, 272(%rsp)
	movq	%r13, 256(%rsp)
	movq	%r12, 192(%rsp)
	vmovdqa	%xmm0, 96(%rsp)
	movq	$1, 112(%rsp)
	movq	$-1, 72(%rsp)
	movq	$1, 176(%rsp)
	movq	$1, 240(%rsp)
	movq	$1, 264(%rsp)
	call	__fnlp_vector_search_core_MOD_l2_database_openmp
	movq	24(%rsp), %rcx
	movq	16(%rsp), %rdx
	leaq	64(%rsp), %rdi
	movq	%rbp, %rsi
	call	__fnlp_topk_core_MOD_topk_smallest@PLT
	addq	$296, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	movq	%r15, %rdi
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmp	free@PLT
	.p2align 4,,10
	.p2align 3
.L112:
	.cfi_restore_state
	movq	$0, 56(%rsp)
	movl	$1, %edi
	jmp	.L108
.L116:
	movq	56(%rsp), %rdx
	leaq	.LC5(%rip), %rsi
	leaq	.LC9(%rip), %rdi
	xorl	%eax, %eax
	call	_gfortran_os_error_at@PLT
.L107:
	leaq	.LC4(%rip), %rdi
	xorl	%eax, %eax
	call	_gfortran_runtime_error@PLT
.L115:
	xorl	%edx, %edx
	movl	$34, %esi
	leaq	.LC3(%rip), %rdi
	call	_gfortran_error_stop_string@PLT
.L114:
	xorl	%edx, %edx
	movl	$21, %esi
	leaq	.LC2(%rip), %rdi
	call	_gfortran_error_stop_string@PLT
	.cfi_endproc
.LFE2:
	.size	__fnlp_vector_search_core_MOD_l2_search_openmp, .-__fnlp_vector_search_core_MOD_l2_search_openmp
	.section	.rodata.str1.8
	.align 8
.LC10:
	.string	"In file 'src/native/cpu/vector_search/vector_search_core.f90', around line 234"
	.text
	.p2align 4
	.globl	__fnlp_vector_search_core_MOD_l2_search_serial
	.type	__fnlp_vector_search_core_MOD_l2_search_serial, @function
__fnlp_vector_search_core_MOD_l2_search_serial:
.LFB3:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	movq	%rsi, %rax
	movl	$1, %r10d
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	movq	%rdx, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$296, %rsp
	.cfi_def_cfa_offset 352
	movq	40(%rsi), %rdx
	movq	56(%rsi), %rbx
	movq	%rcx, 16(%rsp)
	movq	80(%rax), %rcx
	movq	(%rsi), %r12
	movq	%r8, 24(%rsp)
	testq	%rdx, %rdx
	movq	64(%rax), %r13
	movq	56(%rdi), %r14
	cmove	%r10, %rdx
	subq	72(%rax), %rcx
	subq	48(%rsi), %rbx
	leaq	1(%rbx), %rsi
	leaq	1(%rcx), %rbx
	movq	%rcx, %rax
	movq	40(%rdi), %rcx
	movslq	0(%rbp), %r15
	movq	%rsi, (%rsp)
	testq	%rcx, %rcx
	cmove	%r10, %rcx
	subq	48(%rdi), %r14
	xorl	%r11d, %r11d
	testq	%rbx, %rbx
	leaq	1(%r14), %rsi
	movq	(%rdi), %r10
	cmovns	%rbx, %r11
	movq	%rsi, 8(%rsp)
	testl	%r15d, %r15d
	jle	.L129
	movq	%r11, %r14
	cmpq	%r11, %r15
	jg	.L130
	movabsq	$3302829850624, %rsi
	movabsq	$4611686018427387903, %rdi
	movq	$4, 80(%rsp)
	shrq	$63, %rax
	movq	%rsi, 88(%rsp)
	cmpq	%rdi, %rbx
	jg	.L122
	testl	%eax, %eax
	jne	.L127
	leaq	0(,%r11,4), %rax
	movq	%rax, 56(%rsp)
	movq	%rax, %rdi
.L123:
	movq	%r10, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	%rdx, 32(%rsp)
	call	malloc@PLT
	movq	32(%rsp), %rdx
	movq	40(%rsp), %rcx
	testq	%rax, %rax
	movq	%rax, 64(%rsp)
	movq	48(%rsp), %r10
	movq	%rax, %r15
	je	.L131
	movq	8(%rsp), %rax
	vmovdqa	.LC7(%rip), %xmm0
	movq	%rdx, 232(%rsp)
	leaq	192(%rsp), %rsi
	leaq	128(%rsp), %rdi
	movq	%r15, %rdx
	movq	%r14, 120(%rsp)
	movq	%rax, 184(%rsp)
	movq	(%rsp), %rax
	movq	%rcx, 168(%rsp)
	movq	%r10, 128(%rsp)
	movq	%rax, 248(%rsp)
	movq	%rbx, 272(%rsp)
	movq	%r13, 256(%rsp)
	movq	%r12, 192(%rsp)
	vmovdqa	%xmm0, 96(%rsp)
	movq	$1, 112(%rsp)
	movq	$-1, 72(%rsp)
	movq	$1, 176(%rsp)
	movq	$1, 240(%rsp)
	movq	$1, 264(%rsp)
	call	__fnlp_vector_search_core_MOD_l2_database_openmp
	movq	24(%rsp), %rcx
	movq	16(%rsp), %rdx
	leaq	64(%rsp), %rdi
	movq	%rbp, %rsi
	call	__fnlp_topk_core_MOD_topk_smallest@PLT
	addq	$296, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	movq	%r15, %rdi
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmp	free@PLT
	.p2align 4,,10
	.p2align 3
.L127:
	.cfi_restore_state
	movq	$0, 56(%rsp)
	movl	$1, %edi
	jmp	.L123
.L131:
	movq	56(%rsp), %rdx
	leaq	.LC5(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	xorl	%eax, %eax
	call	_gfortran_os_error_at@PLT
.L122:
	leaq	.LC4(%rip), %rdi
	xorl	%eax, %eax
	call	_gfortran_runtime_error@PLT
.L130:
	xorl	%edx, %edx
	movl	$34, %esi
	leaq	.LC3(%rip), %rdi
	call	_gfortran_error_stop_string@PLT
.L129:
	xorl	%edx, %edx
	movl	$21, %esi
	leaq	.LC2(%rip), %rdi
	call	_gfortran_error_stop_string@PLT
	.cfi_endproc
.LFE3:
	.size	__fnlp_vector_search_core_MOD_l2_search_serial, .-__fnlp_vector_search_core_MOD_l2_search_serial
	.p2align 4
	.globl	__fnlp_vector_search_core_MOD_ip_database_serial
	.type	__fnlp_vector_search_core_MOD_ip_database_serial, @function
__fnlp_vector_search_core_MOD_ip_database_serial:
.LFB11:
	.cfi_startproc
	jmp	__fnlp_vector_search_core_MOD_ip_database_openmp
	.cfi_endproc
.LFE11:
	.size	__fnlp_vector_search_core_MOD_ip_database_serial, .-__fnlp_vector_search_core_MOD_ip_database_serial
	.p2align 4
	.globl	__fnlp_vector_search_core_MOD_l2_database_serial
	.type	__fnlp_vector_search_core_MOD_l2_database_serial, @function
__fnlp_vector_search_core_MOD_l2_database_serial:
.LFB13:
	.cfi_startproc
	jmp	__fnlp_vector_search_core_MOD_l2_database_openmp
	.cfi_endproc
.LFE13:
	.size	__fnlp_vector_search_core_MOD_l2_database_serial, .-__fnlp_vector_search_core_MOD_l2_database_serial
	.p2align 4
	.globl	__fnlp_vector_search_core_MOD_inner_product
	.type	__fnlp_vector_search_core_MOD_inner_product, @function
__fnlp_vector_search_core_MOD_inner_product:
.LFB8:
	.cfi_startproc
	movq	40(%rdi), %r10
	movq	56(%rdi), %r11
	movq	%rsi, %r9
	movl	$1, %eax
	movq	(%rdi), %rcx
	movq	%rdx, %r8
	movq	(%r9), %rdx
	testq	%r10, %r10
	cmove	%rax, %r10
	subq	48(%rdi), %r11
	movq	40(%r9), %rdi
	leaq	1(%r11), %rsi
	testq	%rdi, %rdi
	cmove	%rax, %rdi
	xorl	%eax, %eax
	testq	%rsi, %rsi
	cmovs	%rax, %rsi
	movq	56(%r9), %rax
	subq	48(%r9), %rax
	movl	$0, %r9d
	incq	%rax
	cmovs	%r9, %rax
	cmpq	%rsi, %rax
	jne	.L167
	movl	$0x00000000, (%r8)
	testq	%r11, %r11
	js	.L163
	cmpq	$1, %rdi
	jne	.L151
	cmpq	$1, %r10
	jne	.L151
	cmpq	$14, %r11
	jle	.L150
	movq	%rsi, %rdi
	xorl	%eax, %eax
	vxorps	%xmm0, %xmm0, %xmm0
	shrq	$4, %rdi
	salq	$6, %rdi
	.p2align 4,,10
	.p2align 3
.L144:
	vmovups	(%rcx,%rax), %zmm5
	vmulps	(%rdx,%rax), %zmm5, %zmm1
	addq	$64, %rax
	vshufps	$85, %xmm1, %xmm1, %xmm4
	vshufps	$255, %xmm1, %xmm1, %xmm3
	valignd	$7, %ymm1, %ymm1, %ymm2
	vaddss	%xmm1, %xmm0, %xmm0
	vaddss	%xmm4, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm4
	vaddss	%xmm4, %xmm0, %xmm0
	vaddss	%xmm3, %xmm0, %xmm0
	vextractf32x4	$1, %ymm1, %xmm3
	vaddss	%xmm3, %xmm0, %xmm0
	valignd	$5, %ymm1, %ymm1, %ymm3
	vaddss	%xmm3, %xmm0, %xmm0
	valignd	$6, %ymm1, %ymm1, %ymm3
	vextractf32x8	$0x1, %zmm1, %ymm1
	vaddss	%xmm3, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm3
	vaddss	%xmm2, %xmm0, %xmm0
	vshufps	$255, %xmm1, %xmm1, %xmm2
	vaddss	%xmm1, %xmm0, %xmm0
	vaddss	%xmm3, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm3
	vaddss	%xmm3, %xmm0, %xmm0
	vaddss	%xmm2, %xmm0, %xmm0
	vextractf32x4	$1, %ymm1, %xmm2
	vaddss	%xmm2, %xmm0, %xmm0
	valignd	$5, %ymm1, %ymm1, %ymm2
	vaddss	%xmm2, %xmm0, %xmm0
	valignd	$6, %ymm1, %ymm1, %ymm2
	valignd	$7, %ymm1, %ymm1, %ymm1
	vaddss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	cmpq	%rax, %rdi
	jne	.L144
	movq	%rsi, %r9
	andq	$-16, %r9
	leaq	1(%r9), %rax
	cmpq	%r9, %rsi
	je	.L162
.L143:
	movq	%rsi, %rdi
	subq	%r9, %rdi
	leaq	-1(%rdi), %r10
	cmpq	$6, %r10
	jbe	.L146
	vmovups	(%rcx,%r9,4), %ymm7
	vmulps	(%rdx,%r9,4), %ymm7, %ymm1
	movq	%rdi, %r9
	andq	$-8, %r9
	addq	%r9, %rax
	andl	$7, %edi
	vaddss	%xmm1, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm3
	vshufps	$255, %xmm1, %xmm1, %xmm2
	vaddss	%xmm3, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm3
	vaddss	%xmm3, %xmm0, %xmm0
	vaddss	%xmm2, %xmm0, %xmm0
	vextractf32x4	$1, %ymm1, %xmm2
	vaddss	%xmm2, %xmm0, %xmm0
	valignd	$5, %ymm1, %ymm1, %ymm2
	vaddss	%xmm2, %xmm0, %xmm0
	valignd	$6, %ymm1, %ymm1, %ymm2
	valignd	$7, %ymm1, %ymm1, %ymm1
	vaddss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	je	.L162
.L146:
	leaq	-1(%rax), %r9
	vmovss	(%rdx,%r9,4), %xmm7
	leaq	0(,%r9,4), %rdi
	vfmadd231ss	(%rcx,%r9,4), %xmm7, %xmm0
	cmpq	%rax, %rsi
	jle	.L162
	vmovss	4(%rdx,%rdi), %xmm6
	leaq	2(%rax), %r9
	vfmadd231ss	4(%rcx,%rdi), %xmm6, %xmm0
	cmpq	%r9, %rsi
	jl	.L162
	vmovss	8(%rcx,%rdi), %xmm7
	leaq	3(%rax), %r9
	vfmadd231ss	8(%rdx,%rdi), %xmm7, %xmm0
	cmpq	%r9, %rsi
	jl	.L162
	vmovss	12(%rcx,%rdi), %xmm6
	leaq	4(%rax), %r9
	vfmadd231ss	12(%rdx,%rdi), %xmm6, %xmm0
	cmpq	%r9, %rsi
	jl	.L162
	vmovss	16(%rcx,%rdi), %xmm7
	leaq	5(%rax), %r9
	vfmadd231ss	16(%rdx,%rdi), %xmm7, %xmm0
	cmpq	%r9, %rsi
	jl	.L162
	vmovss	20(%rcx,%rdi), %xmm6
	addq	$6, %rax
	vfmadd231ss	20(%rdx,%rdi), %xmm6, %xmm0
	cmpq	%rax, %rsi
	jl	.L162
	vmovss	24(%rcx,%rdi), %xmm7
	vfmadd231ss	24(%rdx,%rdi), %xmm7, %xmm0
	vzeroupper
.L141:
	vmovss	%xmm0, (%r8)
.L163:
	ret
	.p2align 4,,10
	.p2align 3
.L162:
	vzeroupper
	jmp	.L141
	.p2align 4,,10
	.p2align 3
.L151:
	salq	$2, %r10
	salq	$2, %rdi
	vxorps	%xmm0, %xmm0, %xmm0
	movl	$1, %eax
	.p2align 4,,10
	.p2align 3
.L142:
	vmovss	(%rcx), %xmm6
	incq	%rax
	addq	%r10, %rcx
	vfmadd231ss	(%rdx), %xmm6, %xmm0
	addq	%rdi, %rdx
	cmpq	%rax, %rsi
	jge	.L142
	jmp	.L141
.L150:
	xorl	%r9d, %r9d
	vxorps	%xmm0, %xmm0, %xmm0
	movl	$1, %eax
	jmp	.L143
.L167:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	xorl	%edx, %edx
	movl	$54, %esi
	leaq	.LC1(%rip), %rdi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	call	_gfortran_error_stop_string@PLT
	.cfi_endproc
.LFE8:
	.size	__fnlp_vector_search_core_MOD_inner_product, .-__fnlp_vector_search_core_MOD_inner_product
	.p2align 4
	.globl	__fnlp_vector_search_core_MOD_squared_l2_distance
	.type	__fnlp_vector_search_core_MOD_squared_l2_distance, @function
__fnlp_vector_search_core_MOD_squared_l2_distance:
.LFB9:
	.cfi_startproc
	movq	40(%rdi), %r10
	movq	56(%rdi), %r11
	movq	%rsi, %r9
	movl	$1, %eax
	movq	(%rdi), %rcx
	movq	%rdx, %r8
	movq	(%r9), %rdx
	testq	%r10, %r10
	cmove	%rax, %r10
	subq	48(%rdi), %r11
	movq	40(%r9), %rdi
	leaq	1(%r11), %rsi
	testq	%rdi, %rdi
	cmove	%rax, %rdi
	xorl	%eax, %eax
	testq	%rsi, %rsi
	cmovs	%rax, %rsi
	movq	56(%r9), %rax
	subq	48(%r9), %rax
	movl	$0, %r9d
	incq	%rax
	cmovs	%r9, %rax
	cmpq	%rsi, %rax
	jne	.L201
	movl	$0x00000000, (%r8)
	testq	%r11, %r11
	js	.L197
	cmpq	$1, %rdi
	jne	.L185
	cmpq	$1, %r10
	jne	.L185
	cmpq	$14, %r11
	jle	.L184
	movq	%rsi, %rdi
	xorl	%eax, %eax
	vxorps	%xmm0, %xmm0, %xmm0
	shrq	$4, %rdi
	salq	$6, %rdi
	.p2align 4,,10
	.p2align 3
.L178:
	vmovups	(%rcx,%rax), %zmm5
	vsubps	(%rdx,%rax), %zmm5, %zmm1
	addq	$64, %rax
	vmulps	%zmm1, %zmm1, %zmm1
	vshufps	$85, %xmm1, %xmm1, %xmm4
	vshufps	$255, %xmm1, %xmm1, %xmm3
	valignd	$7, %ymm1, %ymm1, %ymm2
	vaddss	%xmm1, %xmm0, %xmm0
	vaddss	%xmm4, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm4
	vaddss	%xmm4, %xmm0, %xmm0
	vaddss	%xmm3, %xmm0, %xmm0
	vextractf32x4	$1, %ymm1, %xmm3
	vaddss	%xmm3, %xmm0, %xmm0
	valignd	$5, %ymm1, %ymm1, %ymm3
	vaddss	%xmm3, %xmm0, %xmm0
	valignd	$6, %ymm1, %ymm1, %ymm3
	vextractf32x8	$0x1, %zmm1, %ymm1
	vaddss	%xmm3, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm3
	vaddss	%xmm2, %xmm0, %xmm0
	vshufps	$255, %xmm1, %xmm1, %xmm2
	vaddss	%xmm1, %xmm0, %xmm0
	vaddss	%xmm3, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm3
	vaddss	%xmm3, %xmm0, %xmm0
	vaddss	%xmm2, %xmm0, %xmm0
	vextractf32x4	$1, %ymm1, %xmm2
	vaddss	%xmm2, %xmm0, %xmm0
	valignd	$5, %ymm1, %ymm1, %ymm2
	vaddss	%xmm2, %xmm0, %xmm0
	valignd	$6, %ymm1, %ymm1, %ymm2
	valignd	$7, %ymm1, %ymm1, %ymm1
	vaddss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	cmpq	%rax, %rdi
	jne	.L178
	movq	%rsi, %r9
	andq	$-16, %r9
	leaq	1(%r9), %rax
	cmpq	%r9, %rsi
	je	.L196
.L177:
	movq	%rsi, %rdi
	subq	%r9, %rdi
	leaq	-1(%rdi), %r10
	cmpq	$6, %r10
	jbe	.L180
	vmovups	(%rcx,%r9,4), %ymm6
	vsubps	(%rdx,%r9,4), %ymm6, %ymm1
	movq	%rdi, %r9
	andq	$-8, %r9
	addq	%r9, %rax
	andl	$7, %edi
	vmulps	%ymm1, %ymm1, %ymm1
	vaddss	%xmm1, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm3
	vshufps	$255, %xmm1, %xmm1, %xmm2
	vaddss	%xmm3, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm3
	vaddss	%xmm3, %xmm0, %xmm0
	vaddss	%xmm2, %xmm0, %xmm0
	vextractf32x4	$1, %ymm1, %xmm2
	vaddss	%xmm2, %xmm0, %xmm0
	valignd	$5, %ymm1, %ymm1, %ymm2
	vaddss	%xmm2, %xmm0, %xmm0
	valignd	$6, %ymm1, %ymm1, %ymm2
	valignd	$7, %ymm1, %ymm1, %ymm1
	vaddss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	je	.L196
.L180:
	leaq	-1(%rax), %r9
	vmovss	(%rcx,%r9,4), %xmm1
	leaq	0(,%r9,4), %rdi
	vsubss	(%rdx,%r9,4), %xmm1, %xmm1
	vfmadd231ss	%xmm1, %xmm1, %xmm0
	cmpq	%rax, %rsi
	jle	.L196
	vmovss	4(%rcx,%rdi), %xmm1
	leaq	2(%rax), %r9
	vsubss	4(%rdx,%rdi), %xmm1, %xmm1
	vfmadd231ss	%xmm1, %xmm1, %xmm0
	cmpq	%r9, %rsi
	jl	.L196
	vmovss	8(%rcx,%rdi), %xmm1
	leaq	3(%rax), %r9
	vsubss	8(%rdx,%rdi), %xmm1, %xmm1
	vfmadd231ss	%xmm1, %xmm1, %xmm0
	cmpq	%r9, %rsi
	jl	.L196
	vmovss	12(%rcx,%rdi), %xmm1
	leaq	4(%rax), %r9
	vsubss	12(%rdx,%rdi), %xmm1, %xmm1
	vfmadd231ss	%xmm1, %xmm1, %xmm0
	cmpq	%r9, %rsi
	jl	.L196
	vmovss	16(%rcx,%rdi), %xmm1
	leaq	5(%rax), %r9
	vsubss	16(%rdx,%rdi), %xmm1, %xmm1
	vfmadd231ss	%xmm1, %xmm1, %xmm0
	cmpq	%r9, %rsi
	jl	.L196
	vmovss	20(%rcx,%rdi), %xmm1
	addq	$6, %rax
	vsubss	20(%rdx,%rdi), %xmm1, %xmm1
	vfmadd231ss	%xmm1, %xmm1, %xmm0
	cmpq	%rax, %rsi
	jl	.L196
	vmovss	24(%rcx,%rdi), %xmm1
	vsubss	24(%rdx,%rdi), %xmm1, %xmm1
	vfmadd231ss	%xmm1, %xmm1, %xmm0
	vzeroupper
.L175:
	vmovss	%xmm0, (%r8)
.L197:
	ret
	.p2align 4,,10
	.p2align 3
.L196:
	vzeroupper
	jmp	.L175
	.p2align 4,,10
	.p2align 3
.L185:
	salq	$2, %r10
	salq	$2, %rdi
	vxorps	%xmm0, %xmm0, %xmm0
	movl	$1, %eax
	.p2align 4,,10
	.p2align 3
.L176:
	vmovss	(%rcx), %xmm1
	incq	%rax
	addq	%r10, %rcx
	vsubss	(%rdx), %xmm1, %xmm1
	addq	%rdi, %rdx
	vfmadd231ss	%xmm1, %xmm1, %xmm0
	cmpq	%rax, %rsi
	jge	.L176
	jmp	.L175
.L184:
	xorl	%r9d, %r9d
	vxorps	%xmm0, %xmm0, %xmm0
	movl	$1, %eax
	jmp	.L177
.L201:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	xorl	%edx, %edx
	movl	$54, %esi
	leaq	.LC1(%rip), %rdi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	call	_gfortran_error_stop_string@PLT
	.cfi_endproc
.LFE9:
	.size	__fnlp_vector_search_core_MOD_squared_l2_distance, .-__fnlp_vector_search_core_MOD_squared_l2_distance
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC7:
	.quad	4
	.quad	1
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
