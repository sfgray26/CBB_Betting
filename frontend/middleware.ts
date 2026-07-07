import { NextRequest, NextResponse } from 'next/server'

export function middleware(request: NextRequest) {
  const apiKey = request.cookies.get('cbb_api_key')?.value
  const isLoginPage = request.nextUrl.pathname === '/login'
  const url = request.nextUrl

  // Unauthenticated users: redirect to login with return URL
  if (!apiKey && !isLoginPage) {
    const redirectUrl = url.pathname + url.search
    const loginUrl = new URL('/login', request.url)
    loginUrl.searchParams.set('redirect', redirectUrl)
    return NextResponse.redirect(loginUrl)
  }

  // Authenticated users on login page: redirect to intended destination
  if (apiKey && isLoginPage) {
    const redirectParam = url.searchParams.get('redirect')
    const targetUrl = redirectParam && redirectParam.startsWith('/')
      ? new URL(redirectParam, request.url)
      : new URL('/war-room', request.url)
    return NextResponse.redirect(targetUrl)
  }

  return NextResponse.next()
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico|manifest.json|icons/).*)'],
}
