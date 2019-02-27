// Edge

template< class VertInfo, class EdgeInfo, class Settings > bool
	Edge< VertInfo,EdgeInfo,Settings >::isEnd( Vertex< VertInfo,EdgeInfo,Settings > *v )
{
	if(!v) return false;
	return vert[0].vert == v || vert[1].vert == v;
}

template< class VertInfo, class EdgeInfo, class Settings > Vertex< VertInfo,EdgeInfo,Settings >
	*Edge< VertInfo,EdgeInfo,Settings >::getEnd( Vertex< VertInfo,EdgeInfo,Settings > *v )
{
	koalaAssert( v,GraphExcNullVert );
	if (vert[0].vert == v) return vert[1].vert;
	if (vert[1].vert == v) return vert[0].vert;
	return NULL;
}

template< class VertInfo, class EdgeInfo, class Settings >
	EdgeDirection Edge< VertInfo,EdgeInfo,Settings >::getDir( Vertex< VertInfo,EdgeInfo,Settings > *v )
{
	if (!isEnd( v )) return EdNone;
	switch (type)
	{
		case Loop: return EdLoop;
		case Undirected: return EdUndir;
	}
	return (vert[V_out].vert == v) ? EdDirOut : EdDirIn;
}

